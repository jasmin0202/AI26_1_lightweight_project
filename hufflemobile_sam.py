import os, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM requires separate first_step and second_step calls")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)


class ShuffleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.stride = stride
        branch_ch = out_ch // 2
        if stride == 1:
            self.branch2 = nn.Sequential(
                nn.Conv2d(branch_ch, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
                nn.Conv2d(branch_ch, branch_ch, 3, 1, 1, groups=branch_ch, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.Conv2d(branch_ch, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
                nn.Conv2d(branch_ch, branch_ch, 3, stride, 1, groups=branch_ch, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.Conv2d(branch_ch, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            return channel_shuffle(torch.cat([x1, self.branch2(x2)], dim=1), 2)
        return channel_shuffle(torch.cat([self.branch1(x), self.branch2(x)], dim=1), 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=200, dropout=0.2):
        super().__init__()
        ch = [32, 176, 352, 704, 1024]

        self.stem = nn.Sequential(
            nn.Conv2d(3, ch[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch[0]), nn.ReLU(inplace=True),
        )
        self.stage2 = self._make_stage(ch[0], ch[1], n=4)
        self.stage3 = self._make_stage(ch[1], ch[2], n=8)
        self.stage4 = self._make_stage(ch[2], ch[3], n=4)
        self.conv5 = nn.Sequential(
            nn.Conv2d(ch[3], ch[4], 1, bias=False),
            nn.BatchNorm2d(ch[4]), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch[4], num_classes))

    def _make_stage(self, in_ch, out_ch, n):
        layers = [ShuffleBlock(in_ch, out_ch, stride=2)]
        for _ in range(n-1):
            layers.append(ShuffleBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return self.fc(torch.flatten(self.gap(x), 1))


def get_mobilenet(num_classes=200):
    m = mobilenet_v3_large(weights=None, num_classes=num_classes, reduced_tail=True)
    m.features[0][0] = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
    return m


class EnsembleModel(nn.Module):
    def __init__(self, m1: nn.Module, m2: nn.Module):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.m1(x) + self.m2(x)) * 0.5


class TinyImageNetDataset(Dataset):
    def __init__(self, root, is_val=False, transform=None):
        self.transform = transform
        self.samples = []

        if is_val:
            self.img_dir = os.path.join(root, 'public_val', 'images')
            lbl_path = os.path.join(root, 'public_val', 'labels.txt')
            with open(lbl_path) as f:
                for line in f:
                    if line.strip():
                        fn, cl = line.strip().split('\t')
                        self.samples.append((os.path.join(self.img_dir, fn), int(cl)))
        else:
            wnids_path = os.path.join(root, 'wnids.txt')
            with open(wnids_path) as f:
                wnids = [line.strip() for line in f if line.strip()]
            class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

            self.train_dir = os.path.join(root, 'train')
            for wnid in wnids:
                images_dir = os.path.join(self.train_dir, wnid, 'images')
                if os.path.exists(images_dir):
                    cl_idx = class_to_idx[wnid]
                    for fn in os.listdir(images_dir):
                        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(images_dir, fn), cl_idx))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, lb = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, lb


def train_single_model_sam(model, train_loader, val_loader, name, epochs=2):
    print(f"\n>> Starting SAM Fine-Tuning for [{name}]", flush=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    base_opt = optim.AdamW
    optimizer = SAM(model.parameters(), base_opt, rho=0.05, lr=1e-5, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            criterion(model(imgs), labels).backward()
            optimizer.second_step(zero_grad=True)
            total_loss += loss.item()

        model.eval()
        cor = tot = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                cor += (model(imgs).argmax(1) == labels).sum().item()
                tot += imgs.size(0)
        print(f"[{name}] Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {cor/tot*100:.2f}%", flush=True)
    return model


def main():
    print("==========================================", flush=True)
    print("?? Accurate SAM Tuning & Ensemble Pipe Active", flush=True)
    print("==========================================", flush=True)

    train_tf = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_root = './student_data'
    train_ds = TinyImageNetDataset(data_root, is_val=False, transform=train_tf)
    val_ds = TinyImageNetDataset(data_root, is_val=True, transform=val_tf)

    print(f">> Total Train Images Found: {len(train_ds):,}", flush=True)
    print(f">> Total Val Images Found: {len(val_ds):,}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    m1 = ShuffleNetV2(200, dropout=0.2).to(DEVICE)
    m2 = get_mobilenet(200).to(DEVICE)

    p1 = sum(p.numel() for p in m1.parameters())
    p2 = sum(p.numel() for p in m2.parameters())
    print(f">> Total Integrated Params: {p1 + p2:,} (Limit: 5,000,000)", flush=True)
    assert p1 + p2 <= 5_000_000, "?? Error: Combined parameters exceed 5M!"

    m1.load_state_dict(torch.load('best_shufflenet_kd_pure.pth', map_location=DEVICE, weights_only=True))
    m2.load_state_dict(torch.load('best_mobilenet_kd_pure.pth', map_location=DEVICE, weights_only=True))

    m1 = train_single_model_sam(m1, train_loader, val_loader, "ShuffleNetV2", epochs=2)
    m2 = train_single_model_sam(m2, train_loader, val_loader, "MobileNetV3", epochs=2)

    m1.eval(); m2.eval()
    ens = EnsembleModel(m1, m2)

    cor = tot = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            cor += (ens(imgs).argmax(1) == labels).sum().item()
            tot += imgs.size(0)
    print(f"\n>> ?? Final Post-SAM Ensemble Val Acc: {cor/tot*100:.2f}%", flush=True)

    m1.cpu(); m2.cpu()
    ens_cpu = EnsembleModel(m1, m2).eval()
    dummy = torch.randn(1, 3, 64, 64)
    traced = torch.jit.trace(ens_cpu, dummy)

    output_filename = 'my_model_sam_ensemble_4.pt'
    torch.jit.save(traced, output_filename)
    print(f'>> [SUCCESS] Exported SAM-Tuned Ensemble file to: {output_filename}', flush=True)


if __name__ == '__main__':
    main()