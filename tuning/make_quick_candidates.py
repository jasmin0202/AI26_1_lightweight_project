#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_quick_candidates.py

Purpose:
- Load two mini-model .pth files:
    1) best_weights_auto_resnet.pth
    2) best_weights_auto_mobilenet.pth
- Evaluate several legal inference-only ensemble variants on public_val.
- Save top candidates as TorchScript .pt files:
    newtest_0.pt, newtest_1.pt, ...

This script does NOT train. It only evaluates and exports candidate models.

Recommended run:
    python make_quick_candidates.py --device cuda:1 --prefix newtest

With TTA disabled:
    python make_quick_candidates.py --device cuda:1 --no_tta --prefix newtest

If DataLoader hangs:
    python make_quick_candidates.py --device cuda:1 --num_workers 0 --prefix newtest
"""

import os
import csv
import json
import math
import argparse
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image


# -------------------------
# Model definitions
# Must match your training code.
# -------------------------

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class TinyResNet18(nn.Module):
    def __init__(self, num_classes=200, width_mult=0.48, dropout=0.2):
        super().__init__()

        ch1 = int(64 * width_mult)
        ch2 = int(128 * width_mult)
        ch3 = int(256 * width_mult)
        ch4 = int(512 * width_mult)

        self.stem = nn.Sequential(
            nn.Conv2d(3, ch1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(ch1, ch1, blocks=2, stride=1)
        self.layer2 = self._make_layer(ch1, ch2, blocks=2, stride=2)
        self.layer3 = self._make_layer(ch2, ch3, blocks=2, stride=2)
        self.layer4 = self._make_layer(ch3, ch4, blocks=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(ch4, num_classes)
        )

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_mobilenet_v3(num_classes=200):
    # torchvision API differs by version.
    try:
        model = mobilenet_v3_large(weights=None, num_classes=num_classes, reduced_tail=True)
    except TypeError:
        model = mobilenet_v3_large(pretrained=False, num_classes=num_classes, reduced_tail=True)

    model.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    return model


# -------------------------
# Dataset
# -------------------------

class TinyImageNetVal(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir = os.path.join(root, "images")
        self.transform = transform
        self.samples = []

        label_path = os.path.join(root, "labels.txt")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"labels.txt not found: {label_path}")

        with open(label_path, "r") as f:
            for line in f:
                if line.strip():
                    fname, cls = line.strip().split("\t")
                    self.samples.append((fname, int(cls)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# -------------------------
# Ensemble wrappers for export
# -------------------------

class WeightedLogitEnsemble(nn.Module):
    def __init__(self, resnet, mobilenet, w_resnet: float):
        super().__init__()
        self.resnet = resnet
        self.mobilenet = mobilenet
        self.w_resnet = float(w_resnet)
        self.w_mobilenet = float(1.0 - w_resnet)

    def forward_once(self, x):
        return self.w_resnet * self.resnet(x) + self.w_mobilenet * self.mobilenet(x)

    def forward(self, x):
        return self.forward_once(x)


class WeightedLogitEnsembleTTA(nn.Module):
    def __init__(self, resnet, mobilenet, w_resnet: float):
        super().__init__()
        self.resnet = resnet
        self.mobilenet = mobilenet
        self.w_resnet = float(w_resnet)
        self.w_mobilenet = float(1.0 - w_resnet)

    def forward_once(self, x):
        return self.w_resnet * self.resnet(x) + self.w_mobilenet * self.mobilenet(x)

    def forward(self, x):
        y1 = self.forward_once(x)
        y2 = self.forward_once(torch.flip(x, dims=[3]))
        return (y1 + y2) * 0.5


class TempLogitEnsemble(nn.Module):
    def __init__(self, resnet, mobilenet, w_resnet: float, t_resnet: float, t_mobilenet: float):
        super().__init__()
        self.resnet = resnet
        self.mobilenet = mobilenet
        self.w_resnet = float(w_resnet)
        self.w_mobilenet = float(1.0 - w_resnet)
        self.t_resnet = float(t_resnet)
        self.t_mobilenet = float(t_mobilenet)

    def forward_once(self, x):
        a = self.resnet(x) / self.t_resnet
        b = self.mobilenet(x) / self.t_mobilenet
        return self.w_resnet * a + self.w_mobilenet * b

    def forward(self, x):
        return self.forward_once(x)


class TempLogitEnsembleTTA(nn.Module):
    def __init__(self, resnet, mobilenet, w_resnet: float, t_resnet: float, t_mobilenet: float):
        super().__init__()
        self.resnet = resnet
        self.mobilenet = mobilenet
        self.w_resnet = float(w_resnet)
        self.w_mobilenet = float(1.0 - w_resnet)
        self.t_resnet = float(t_resnet)
        self.t_mobilenet = float(t_mobilenet)

    def forward_once(self, x):
        a = self.resnet(x) / self.t_resnet
        b = self.mobilenet(x) / self.t_mobilenet
        return self.w_resnet * a + self.w_mobilenet * b

    def forward(self, x):
        y1 = self.forward_once(x)
        y2 = self.forward_once(torch.flip(x, dims=[3]))
        return (y1 + y2) * 0.5


class ProbEnsemble(nn.Module):
    """
    Returns log-probabilities as pseudo-logits.
    Argmax is what matters for top-1 accuracy.
    """
    def __init__(self, resnet, mobilenet, w_resnet: float, t_resnet: float, t_mobilenet: float):
        super().__init__()
        self.resnet = resnet
        self.mobilenet = mobilenet
        self.w_resnet = float(w_resnet)
        self.w_mobilenet = float(1.0 - w_resnet)
        self.t_resnet = float(t_resnet)
        self.t_mobilenet = float(t_mobilenet)

    def forward_once(self, x):
        pa = F.softmax(self.resnet(x) / self.t_resnet, dim=1)
        pb = F.softmax(self.mobilenet(x) / self.t_mobilenet, dim=1)
        p = self.w_resnet * pa + self.w_mobilenet * pb
        return torch.log(torch.clamp(p, min=1e-12))

    def forward(self, x):
        return self.forward_once(x)


class ProbEnsembleTTA(nn.Module):
    """
    Returns log-probabilities as pseudo-logits.
    """
    def __init__(self, resnet, mobilenet, w_resnet: float, t_resnet: float, t_mobilenet: float):
        super().__init__()
        self.resnet = resnet
        self.mobilenet = mobilenet
        self.w_resnet = float(w_resnet)
        self.w_mobilenet = float(1.0 - w_resnet)
        self.t_resnet = float(t_resnet)
        self.t_mobilenet = float(t_mobilenet)

    def forward_once(self, x):
        pa = F.softmax(self.resnet(x) / self.t_resnet, dim=1)
        pb = F.softmax(self.mobilenet(x) / self.t_mobilenet, dim=1)
        p = self.w_resnet * pa + self.w_mobilenet * pb
        return torch.log(torch.clamp(p, min=1e-12))

    def forward(self, x):
        y1 = self.forward_once(x)
        y2 = self.forward_once(torch.flip(x, dims=[3]))
        return (y1 + y2) * 0.5


class SingleModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class SingleModelTTAWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y1 = self.model(x)
        y2 = self.model(torch.flip(x, dims=[3]))
        return (y1 + y2) * 0.5


# -------------------------
# Utility
# -------------------------

@dataclass
class Candidate:
    name: str
    acc: float
    kind: str
    w_resnet: Optional[float] = None
    t_resnet: Optional[float] = None
    t_mobilenet: Optional[float] = None
    use_tta: bool = False
    output_pt: Optional[str] = None
    params: Optional[int] = None
    file_size_mb: Optional[float] = None


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(state, dict):
        return state
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def load_state_dict_flexible(model: nn.Module, path: str, model_name: str):
    print(f"Loading {model_name} weights: {path}")
    state = torch.load(path, map_location="cpu")
    state = strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing or unexpected:
        print(f"[WARN] {model_name} load_state_dict strict=False")
        print(f"       missing keys   : {len(missing)}")
        print(f"       unexpected keys: {len(unexpected)}")

        # Try strict load to fail loudly if it is seriously mismatched.
        try:
            model.load_state_dict(state, strict=True)
            print(f"[INFO] {model_name} strict=True actually passed on retry.")
        except Exception as e:
            print(f"[WARN] strict=True failed for {model_name}. Continuing with strict=False.")
            print(f"       error: {e}")
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean().item()


def make_loader(data_root: str, batch_size: int, num_workers: int):
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    val_ds = TinyImageNetVal(os.path.join(data_root, "public_val"), transform=val_tf)

    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader


@torch.no_grad()
def collect_logits(
    resnet: nn.Module,
    mobilenet: nn.Module,
    loader: DataLoader,
    device: torch.device,
    collect_tta: bool,
):
    resnet.eval()
    mobilenet.eval()

    r_list, m_list, y_list = [], [], []
    r_flip_list, m_flip_list = [], []

    total_seen = 0
    for step, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        r = resnet(imgs)
        m = mobilenet(imgs)

        r_list.append(r.cpu())
        m_list.append(m.cpu())
        y_list.append(labels.cpu())

        if collect_tta:
            imgs_flip = torch.flip(imgs, dims=[3])
            r_flip = resnet(imgs_flip)
            m_flip = mobilenet(imgs_flip)
            r_flip_list.append(r_flip.cpu())
            m_flip_list.append(m_flip.cpu())

        total_seen += imgs.size(0)
        if step == 0 or (step + 1) % 10 == 0:
            print(f"  collected logits: {total_seen} images")

    out = {
        "resnet": torch.cat(r_list, dim=0),
        "mobilenet": torch.cat(m_list, dim=0),
        "labels": torch.cat(y_list, dim=0),
    }

    if collect_tta:
        out["resnet_flip"] = torch.cat(r_flip_list, dim=0)
        out["mobilenet_flip"] = torch.cat(m_flip_list, dim=0)

    return out


def search_weighted(logits_r, logits_m, labels, w_grid: List[float], name_prefix="weighted") -> Candidate:
    best = Candidate(name=name_prefix, acc=-1.0, kind="weighted_logit")

    for w in w_grid:
        logits = w * logits_r + (1.0 - w) * logits_m
        acc = accuracy_from_logits(logits, labels)

        if acc > best.acc:
            best = Candidate(
                name=f"{name_prefix}_w{w:.4f}",
                acc=acc,
                kind="weighted_logit",
                w_resnet=w,
                use_tta=False,
            )

    return best


def search_temp_logit(logits_r, logits_m, labels, w_grid: List[float], t_grid: List[float], name_prefix="temp_logit") -> Candidate:
    best = Candidate(name=name_prefix, acc=-1.0, kind="temp_logit")

    for tr in t_grid:
        r_scaled = logits_r / tr
        for tm in t_grid:
            m_scaled = logits_m / tm
            for w in w_grid:
                logits = w * r_scaled + (1.0 - w) * m_scaled
                acc = accuracy_from_logits(logits, labels)

                if acc > best.acc:
                    best = Candidate(
                        name=f"{name_prefix}_w{w:.4f}_tr{tr:.3f}_tm{tm:.3f}",
                        acc=acc,
                        kind="temp_logit",
                        w_resnet=w,
                        t_resnet=tr,
                        t_mobilenet=tm,
                        use_tta=False,
                    )

    return best


def search_prob_ensemble(logits_r, logits_m, labels, w_grid: List[float], t_grid: List[float], name_prefix="prob") -> Candidate:
    best = Candidate(name=name_prefix, acc=-1.0, kind="prob")

    for tr in t_grid:
        pr = F.softmax(logits_r / tr, dim=1)
        for tm in t_grid:
            pm = F.softmax(logits_m / tm, dim=1)
            for w in w_grid:
                probs = w * pr + (1.0 - w) * pm
                acc = accuracy_from_logits(probs, labels)

                if acc > best.acc:
                    best = Candidate(
                        name=f"{name_prefix}_w{w:.4f}_tr{tr:.3f}_tm{tm:.3f}",
                        acc=acc,
                        kind="prob",
                        w_resnet=w,
                        t_resnet=tr,
                        t_mobilenet=tm,
                        use_tta=False,
                    )

    return best


def eval_candidate_from_cached_logits(c: Candidate, cache: Dict[str, torch.Tensor]) -> Candidate:
    labels = cache["labels"]
    r = cache["resnet"]
    m = cache["mobilenet"]

    if c.use_tta:
        r = (cache["resnet"] + cache["resnet_flip"]) * 0.5
        m = (cache["mobilenet"] + cache["mobilenet_flip"]) * 0.5

    if c.kind == "resnet":
        logits = r
    elif c.kind == "mobilenet":
        logits = m
    elif c.kind == "weighted_logit":
        logits = c.w_resnet * r + (1.0 - c.w_resnet) * m
    elif c.kind == "temp_logit":
        logits = c.w_resnet * (r / c.t_resnet) + (1.0 - c.w_resnet) * (m / c.t_mobilenet)
    elif c.kind == "prob":
        pr = F.softmax(r / c.t_resnet, dim=1)
        pm = F.softmax(m / c.t_mobilenet, dim=1)
        logits = c.w_resnet * pr + (1.0 - c.w_resnet) * pm
    else:
        raise ValueError(f"Unknown candidate kind: {c.kind}")

    c.acc = accuracy_from_logits(logits, labels)
    return c


def build_export_model(c: Candidate, resnet: nn.Module, mobilenet: nn.Module) -> nn.Module:
    # Build fresh CPU modules outside before calling this if desired.
    if c.kind == "resnet":
        if c.use_tta:
            return SingleModelTTAWrapper(resnet)
        return SingleModelWrapper(resnet)

    if c.kind == "mobilenet":
        if c.use_tta:
            return SingleModelTTAWrapper(mobilenet)
        return SingleModelWrapper(mobilenet)

    if c.kind == "weighted_logit":
        if c.use_tta:
            return WeightedLogitEnsembleTTA(resnet, mobilenet, c.w_resnet)
        return WeightedLogitEnsemble(resnet, mobilenet, c.w_resnet)

    if c.kind == "temp_logit":
        if c.use_tta:
            return TempLogitEnsembleTTA(resnet, mobilenet, c.w_resnet, c.t_resnet, c.t_mobilenet)
        return TempLogitEnsemble(resnet, mobilenet, c.w_resnet, c.t_resnet, c.t_mobilenet)

    if c.kind == "prob":
        if c.use_tta:
            return ProbEnsembleTTA(resnet, mobilenet, c.w_resnet, c.t_resnet, c.t_mobilenet)
        return ProbEnsemble(resnet, mobilenet, c.w_resnet, c.t_resnet, c.t_mobilenet)

    raise ValueError(f"Unknown candidate kind: {c.kind}")


def export_candidate(
    c: Candidate,
    output_path: str,
    resnet_pth: str,
    mobilenet_pth: str,
    dropout: float,
):
    # Rebuild CPU models so exported file is independent from GPU objects.
    resnet = TinyResNet18(num_classes=200, width_mult=0.48, dropout=dropout)
    mobilenet = get_mobilenet_v3(num_classes=200)

    load_state_dict_flexible(resnet, resnet_pth, "ResNet-export")
    load_state_dict_flexible(mobilenet, mobilenet_pth, "MobileNet-export")

    resnet.eval()
    mobilenet.eval()

    model = build_export_model(c, resnet, mobilenet)
    model.eval()

    params = count_params(model)
    if params > 5_000_000:
        raise RuntimeError(f"Param limit exceeded for {c.name}: {params:,}")

    dummy = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        out = model(dummy)

    if tuple(out.shape) != (1, 200):
        raise RuntimeError(f"Bad output shape for {c.name}: {tuple(out.shape)}")

    traced = torch.jit.trace(model, dummy)
    torch.jit.save(traced, output_path)

    # Reload sanity check.
    reloaded = torch.jit.load(output_path, map_location="cpu")
    reloaded.eval()
    with torch.no_grad():
        out2 = reloaded(dummy)

    if tuple(out2.shape) != (1, 200):
        raise RuntimeError(f"Reloaded bad output shape for {c.name}: {tuple(out2.shape)}")

    c.output_pt = output_path
    c.params = params
    c.file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return c


def save_manifest(candidates: List[Candidate], out_csv: str, out_json: str):
    rows = [asdict(c) for c in candidates]

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./student_data")
    parser.add_argument("--resnet_pth", type=str, default="./best_weights_auto_resnet.pth")
    parser.add_argument("--mobilenet_pth", type=str, default="./best_weights_auto_mobilenet.pth")
    parser.add_argument("--prefix", type=str, default="newtest")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--save_top_k", type=int, default=6)
    parser.add_argument("--no_tta", action="store_true")

    parser.add_argument(
        "--w_grid",
        type=str,
        default="0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00",
        help="Comma-separated ResNet weights."
    )
    parser.add_argument(
        "--w_fine_center",
        type=float,
        default=-1.0,
        help="If >=0, also search around this center with step 0.01."
    )
    parser.add_argument(
        "--t_grid",
        type=str,
        default="0.60,0.70,0.80,0.90,1.00,1.10,1.20,1.35,1.50,1.75,2.00,2.50,3.00",
        help="Comma-separated temperature values."
    )

    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("==============================")
    print("Quick candidate maker")
    print("==============================")
    print("device       :", device)
    print("data_root    :", args.data_root)
    print("resnet_pth   :", args.resnet_pth)
    print("mobilenet_pth:", args.mobilenet_pth)
    print("prefix       :", args.prefix)
    print("run_tta      :", not args.no_tta)
    print("==============================")

    # Build models.
    resnet = TinyResNet18(num_classes=200, width_mult=0.48, dropout=args.dropout)
    mobilenet = get_mobilenet_v3(num_classes=200)

    load_state_dict_flexible(resnet, args.resnet_pth, "ResNet")
    load_state_dict_flexible(mobilenet, args.mobilenet_pth, "MobileNet")

    p_r = count_params(resnet)
    p_m = count_params(mobilenet)
    print(f"ResNet params   : {p_r:,}")
    print(f"MobileNet params: {p_m:,}")
    print(f"Ensemble params : {p_r + p_m:,}")
    assert p_r + p_m <= 5_000_000, f"Combined params exceeded: {p_r + p_m:,}"

    resnet = resnet.to(device).eval()
    mobilenet = mobilenet.to(device).eval()

    loader = make_loader(args.data_root, args.batch_size, args.num_workers)

    # Collect logits once.
    print("\n[1] Collecting logits")
    cache = collect_logits(resnet, mobilenet, loader, device, collect_tta=(not args.no_tta))
    labels = cache["labels"]

    w_grid = parse_float_list(args.w_grid)
    t_grid = parse_float_list(args.t_grid)

    # Optional fine weight grid.
    if args.w_fine_center >= 0:
        center = args.w_fine_center
        fine = []
        start = max(0.0, center - 0.08)
        end = min(1.0, center + 0.08)
        n = int(round((end - start) / 0.01)) + 1
        for i in range(n):
            fine.append(round(start + i * 0.01, 4))
        w_grid = sorted(set(w_grid + fine))

    print("\n[2] Basic accuracies")
    candidates: List[Candidate] = []

    c_res = Candidate(name="resnet_only", acc=0.0, kind="resnet")
    c_mob = Candidate(name="mobilenet_only", acc=0.0, kind="mobilenet")
    c_half = Candidate(name="half_logit_w0.5000", acc=0.0, kind="weighted_logit", w_resnet=0.5)

    for c in [c_res, c_mob, c_half]:
        c = eval_candidate_from_cached_logits(c, cache)
        candidates.append(c)
        print(f"{c.name:32s} | acc={c.acc * 100:.4f}%")

    print("\n[3] Searching weighted logit ensemble")
    best_weighted = search_weighted(cache["resnet"], cache["mobilenet"], labels, w_grid, name_prefix="best_weighted")
    candidates.append(best_weighted)
    print(f"{best_weighted.name:32s} | acc={best_weighted.acc * 100:.4f}%")

    print("\n[4] Searching temperature-scaled logit ensemble")
    best_temp = search_temp_logit(cache["resnet"], cache["mobilenet"], labels, w_grid, t_grid, name_prefix="best_temp_logit")
    candidates.append(best_temp)
    print(f"{best_temp.name:32s} | acc={best_temp.acc * 100:.4f}%")

    print("\n[5] Searching probability ensemble")
    best_prob = search_prob_ensemble(cache["resnet"], cache["mobilenet"], labels, w_grid, t_grid, name_prefix="best_prob")
    candidates.append(best_prob)
    print(f"{best_prob.name:32s} | acc={best_prob.acc * 100:.4f}%")

    # Evaluate TTA versions of strong candidates.
    if not args.no_tta:
        print("\n[6] Evaluating TTA candidates")
        tta_candidates = []

        # Single-model TTA.
        tta_candidates.append(Candidate(name="resnet_only_tta", acc=0.0, kind="resnet", use_tta=True))
        tta_candidates.append(Candidate(name="mobilenet_only_tta", acc=0.0, kind="mobilenet", use_tta=True))

        # Half TTA.
        tta_candidates.append(Candidate(name="half_logit_w0.5000_tta", acc=0.0, kind="weighted_logit", w_resnet=0.5, use_tta=True))

        # Best variants TTA.
        tta_candidates.append(Candidate(
            name=best_weighted.name + "_tta",
            acc=0.0,
            kind="weighted_logit",
            w_resnet=best_weighted.w_resnet,
            use_tta=True,
        ))

        tta_candidates.append(Candidate(
            name=best_temp.name + "_tta",
            acc=0.0,
            kind="temp_logit",
            w_resnet=best_temp.w_resnet,
            t_resnet=best_temp.t_resnet,
            t_mobilenet=best_temp.t_mobilenet,
            use_tta=True,
        ))

        tta_candidates.append(Candidate(
            name=best_prob.name + "_tta",
            acc=0.0,
            kind="prob",
            w_resnet=best_prob.w_resnet,
            t_resnet=best_prob.t_resnet,
            t_mobilenet=best_prob.t_mobilenet,
            use_tta=True,
        ))

        for c in tta_candidates:
            c = eval_candidate_from_cached_logits(c, cache)
            candidates.append(c)
            print(f"{c.name:32s} | acc={c.acc * 100:.4f}%")

    # Sort by accuracy.
    candidates = sorted(candidates, key=lambda x: x.acc, reverse=True)

    print("\n==============================")
    print("Candidate ranking")
    print("==============================")
    for i, c in enumerate(candidates):
        print(
            f"{i:02d}. {c.name:45s} | "
            f"acc={c.acc * 100:.4f}% | "
            f"kind={c.kind} | "
            f"w={c.w_resnet} | "
            f"Tr={c.t_resnet} | "
            f"Tm={c.t_mobilenet} | "
            f"TTA={c.use_tta}"
        )

    # Export top K.
    print("\n[7] Exporting top candidates")
    exported: List[Candidate] = []
    top_k = min(args.save_top_k, len(candidates))

    for rank in range(top_k):
        c = candidates[rank]
        out_path = f"{args.prefix}_{rank}.pt"
        print(f"\nExport rank {rank}: {c.name} -> {out_path}")
        c = export_candidate(
            c,
            output_path=out_path,
            resnet_pth=args.resnet_pth,
            mobilenet_pth=args.mobilenet_pth,
            dropout=args.dropout,
        )
        exported.append(c)
        print(f"  saved      : {c.output_pt}")
        print(f"  params     : {c.params:,}")
        print(f"  file size  : {c.file_size_mb:.2f} MB")
        print(f"  public_val : {c.acc * 100:.4f}%")

    manifest_csv = f"{args.prefix}_manifest.csv"
    manifest_json = f"{args.prefix}_manifest.json"

    # Save all candidates, not just exported.
    save_manifest(candidates, manifest_csv, manifest_json)

    print("\n==============================")
    print("Done")
    print("==============================")
    print(f"Saved top {top_k} models:")
    for c in exported:
        print(f"  {c.output_pt:20s} | acc={c.acc * 100:.4f}% | {c.name}")

    print(f"\nManifest:")
    print(f"  {manifest_csv}")
    print(f"  {manifest_json}")

    print("\nImportant:")
    print("- newtest_0.pt is the best public_val candidate from this script.")
    print("- Do not blindly submit every file. Prefer candidates with meaningful gain over 0.5 ensemble.")
    print("- If TTA gain is tiny, non-TTA may be safer for server timeout.")


if __name__ == "__main__":
    main()
