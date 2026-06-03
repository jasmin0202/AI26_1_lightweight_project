"""
ConvNeXt-Tiny v4
- ConvNeXt 아키텍처 (5M 이하)
- RandAugment
- CutMix + Mixup
- 300 epoch, CosineAnnealing + warmup
- cuda:0 사용 (비어있는 GPU)
"""

import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image

# ── 재현성 ────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ── 설정 ──────────────────────────────────────────────────────────────────────
CFG = dict(
    data_root    = './student_data',
    batch_size   = 256,
    num_epochs   = 300,
    lr           = 0.1,
    momentum     = 0.9,
    weight_decay = 5e-5,
    label_smooth = 0.1,
    dropout      = 0.2,
    num_workers  = 4,
    cutmix_alpha = 1.0,
    mixup_alpha  = 0.2,
    output_pt    = 'my_model_v4.pt',
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ConvNeXt Block
# ─────────────────────────────────────────────────────────────────────────────
class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block:
    DWConv 7x7 → LayerNorm → Linear(4x expand) → GELU → Linear → drop
    """
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm   = nn.LayerNorm(dim, eps=1e-6)
        self.pw1    = nn.Linear(dim, dim * 4)
        self.act    = nn.GELU()
        self.pw2    = nn.Linear(dim * 4, dim)
        self.gamma  = nn.Parameter(1e-6 * torch.ones(dim))
        self.drop_path_prob = drop_path

    def drop_path(self, x):
        if not self.training or self.drop_path_prob == 0:
            return x
        keep = 1 - self.drop_path_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep
        return x * mask / keep

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)       # (B,C,H,W) → (B,H,W,C)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x * self.gamma
        x = x.permute(0, 3, 1, 2)       # (B,H,W,C) → (B,C,H,W)
        return residual + self.drop_path(x)


class ConvNeXtStage(nn.Module):
    def __init__(self, in_ch, out_ch, depth, downsample=True, drop_path=0.0):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.LayerNorm([in_ch, 1, 1], eps=1e-6),   # channel-wise
            nn.Conv2d(in_ch, out_ch, 2, stride=2),
        ) if downsample else nn.Identity()

        # LayerNorm for channel dim in downsample
        if downsample:
            self.downsample = nn.Sequential(
                nn.GroupNorm(1, in_ch, eps=1e-6),
                nn.Conv2d(in_ch, out_ch, 2, stride=2),
            )

        self.blocks = nn.Sequential(
            *[ConvNeXtBlock(out_ch, drop_path=drop_path) for _ in range(depth)]
        )

    def forward(self, x):
        x = self.downsample(x)
        return self.blocks(x)


class ConvNeXtTiny(nn.Module):
    """
    ConvNeXt-Tiny 축소판 (5M 이하, 64×64 입력)
    dims: [64, 128, 256, 384]
    depths: [2, 2, 4, 2]
    """
    def __init__(self, num_classes=200, dropout=0.2, drop_path=0.1):
        super().__init__()
        dims   = [56, 112, 232, 352]
        depths = [2, 2, 4, 2]

        # Stem: 64×64 → 16×16
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 4, stride=4, padding=0),
            nn.GroupNorm(1, dims[0], eps=1e-6),
        )

        # 스테이지 (다운샘플 없이 시작)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()

        # Stage 0: 16×16, no downsample
        self.stages.append(nn.Sequential(
            *[ConvNeXtBlock(dims[0], drop_path=dp_rates[cur+i]) for i in range(depths[0])]
        ))
        cur += depths[0]

        # Stage 1~3: downsample
        for i in range(1, 4):
            stage = ConvNeXtStage(
                dims[i-1], dims[i], depths[i],
                downsample=True,
                drop_path=dp_rates[cur]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dims[-1], num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = x.mean([-2, -1])          # GlobalAvgPool
        x = self.norm(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Loss
# ─────────────────────────────────────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, classes=200, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.cls = classes
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.cls - 1)
        log_probs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, smooth_val)
            true_dist.scatter_(1, target.unsqueeze(1), confidence)
        return -(true_dist * log_probs).sum(dim=-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────────────────────────────────────
class TinyImageNetVal(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir   = os.path.join(root, 'images')
        self.transform = transform
        self.samples   = []
        with open(os.path.join(root, 'labels.txt')) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                fname, cls = line.split('\t')
                self.samples.append((fname, int(cls)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label


def get_loaders(cfg):
    train_tf = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(
        os.path.join(cfg['data_root'], 'train'), transform=train_tf)
    val_ds = TinyImageNetVal(
        os.path.join(cfg['data_root'], 'public_val'), transform=val_tf)
    print(f'Train: {len(train_ds):,}  Val: {len(val_ds):,}')
    pw = cfg['num_workers'] > 0
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True, persistent_workers=pw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=cfg['num_workers'], pin_memory=True, persistent_workers=pw)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# 4. Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def cutmix(imgs, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    _, _, H, W = imgs.shape
    cx, cy = np.random.randint(W), np.random.randint(H)
    w = int(W * math.sqrt(1 - lam))
    h = int(H * math.sqrt(1 - lam))
    x1, x2 = max(cx - w//2, 0), min(cx + w//2, W)
    y1, y2 = max(cy - h//2, 0), min(cy + h//2, H)
    imgs = imgs.clone()
    imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2-x1)*(y2-y1)/(W*H)
    return imgs, labels, labels[idx], lam

def mixup(imgs, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    imgs = lam * imgs + (1 - lam) * imgs[idx]
    return imgs, labels, labels[idx], lam


# ─────────────────────────────────────────────────────────────────────────────
# 5. Train / Eval
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, cfg):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        r = np.random.rand()
        if r < 0.4:
            imgs, la, lb, lam = cutmix(imgs, labels, cfg['cutmix_alpha'])
        elif r < 0.7:
            imgs, la, lb, lam = mixup(imgs, labels, cfg['mixup_alpha'])
        else:
            la, lb, lam = labels, labels, 1.0
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == la).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total   += imgs.size(0)
    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg = CFG

    model = ConvNeXtTiny(num_classes=200, dropout=cfg['dropout']).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}')
    assert params <= 5_000_000, f'파라미터 초과: {params:,}'
    print('파라미터 OK')

    criterion = LabelSmoothingCE(200, cfg['label_smooth'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=4e-3,
        weight_decay=cfg['weight_decay'],
    )

    # Cosine + warmup 20 epoch
    def lr_lambda(epoch):
        warmup = 20
        if epoch < warmup: return (epoch + 1) / warmup
        progress = (epoch - warmup) / (cfg['num_epochs'] - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda')
    train_loader, val_loader = get_loaders(cfg)

    best_acc = 0.0
    for epoch in range(cfg['num_epochs']):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg)
        val_acc = evaluate(model, val_loader)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        print(f'[{epoch+1:3d}/{cfg["num_epochs"]}] loss={tr_loss:.4f}  tr={tr_acc:.4f}  val={val_acc:.4f}  lr={lr_now:.6f}')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_weights_v4.pth')
            print(f'  ✓ best: {best_acc:.4f}')

    print(f'\n학습 완료. Best val acc: {best_acc:.4f}')

    # TorchScript export
    model.load_state_dict(torch.load('best_weights_v4.pth', map_location='cpu'))
    model.eval().cpu()
    dummy  = torch.randn(1, 3, 64, 64)
    traced = torch.jit.trace(model, dummy)
    torch.jit.save(traced, cfg['output_pt'])

    loaded  = torch.jit.load(cfg['output_pt'], map_location='cpu').eval()
    p_count = sum(p.numel() for p in loaded.parameters())
    with torch.no_grad():
        out = loaded(dummy)
    assert out.shape == (1, 200)
    assert p_count <= 5_000_000
    print(f'✓ Export OK — {p_count:,} params, shape {tuple(out.shape)}')


if __name__ == '__main__':
    main()
