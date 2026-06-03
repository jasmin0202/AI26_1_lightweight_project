"""
EfficientNet v5 — v2 체크포인트 fine-tuning
- AdamW optimizer
- Random Erasing 추가
- Drop Path 추가
- 100 epoch fine-tuning
- cuda:0 사용
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

CFG = dict(
    data_root      = './student_data',
    batch_size     = 256,
    num_epochs     = 100,
    lr             = 1e-3,        # AdamW fine-tuning lr
    weight_decay   = 0.05,        # AdamW 권장값
    label_smooth   = 0.1,
    dropout        = 0.2,
    drop_path      = 0.1,
    num_workers    = 4,
    cutmix_alpha   = 1.0,
    mixup_alpha    = 0.2,
    checkpoint     = 'best_weights_v2.pth',
    output_pt      = 'my_model_v5.pt',
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model (Drop Path 추가된 EfficientNet)
# ─────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, expand_ratio,
                 se_ratio=0.25, drop_path=0.0):
        super().__init__()
        self.use_skip = (stride == 1 and in_ch == out_ch)
        self.drop_path_prob = drop_path
        mid_ch = in_ch * expand_ratio
        pad = (kernel - 1) // 2
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                       nn.BatchNorm2d(mid_ch), nn.SiLU()]
        layers += [nn.Conv2d(mid_ch, mid_ch, kernel, stride, pad, groups=mid_ch, bias=False),
                   nn.BatchNorm2d(mid_ch), nn.SiLU()]
        layers.append(SEBlock(mid_ch, reduction=max(1, int(1 / se_ratio))))
        layers += [nn.Conv2d(mid_ch, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(*layers)

    def drop_path(self, x):
        if not self.training or self.drop_path_prob == 0:
            return x
        keep = 1 - self.drop_path_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device) < keep).float()
        return x * mask / keep

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            return x + self.drop_path(out)
        return out


class EfficientNetB0(nn.Module):
    STAGES = [
        (1, 16,  1, 3, 1),
        (6, 24,  2, 3, 2),
        (6, 40,  2, 5, 2),
        (6, 64,  3, 3, 1),
        (6, 88,  3, 5, 1),
        (6, 144, 4, 5, 2),
        (6, 240, 1, 3, 1),
    ]
    def __init__(self, num_classes=200, dropout=0.2, drop_path=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
        )
        # drop path rate 선형 증가
        total_blocks = sum(n for _, _, n, _, _ in self.STAGES)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, total_blocks)]
        blocks, in_ch, idx = [], 32, 0
        for expand, out_ch, n_layers, k, s in self.STAGES:
            for i in range(n_layers):
                blocks.append(MBConv(in_ch, out_ch, k, s if i == 0 else 1,
                                     expand, drop_path=dp_rates[idx]))
                in_ch = out_ch
                idx += 1
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv2d(240, 960, 1, bias=False),
            nn.BatchNorm2d(960), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(960, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


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
        transforms.RandomErasing(p=0.25),   # Random Erasing 추가
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

    model = EfficientNetB0(num_classes=200, dropout=cfg['dropout'],
                           drop_path=cfg['drop_path']).to(DEVICE)

    # v2 체크포인트 로드
    ckpt = torch.load(cfg['checkpoint'], map_location=DEVICE)
    # drop_path 추가로 키가 일부 다를 수 있어서 strict=False
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f'체크포인트 로드 완료')
    if missing:    print(f'  missing keys: {len(missing)}')
    if unexpected: print(f'  unexpected keys: {len(unexpected)}')

    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}')
    assert params <= 5_000_000, f'파라미터 초과: {params:,}'

    # 초기 val 확인
    _, val_loader = get_loaders(cfg)
    init_val = evaluate(model, val_loader)
    print(f'초기 val acc (v2 체크포인트): {init_val:.4f}')

    train_loader, val_loader = get_loaders(cfg)

    criterion = LabelSmoothingCE(200, cfg['label_smooth'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
    )

    # Cosine annealing (warmup 5 epoch)
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup: return (epoch + 1) / warmup
        progress = (epoch - warmup) / (cfg['num_epochs'] - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda')

    best_acc = init_val
    for epoch in range(cfg['num_epochs']):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg)
        val_acc = evaluate(model, val_loader)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        print(f'[{epoch+1:3d}/{cfg["num_epochs"]}] loss={tr_loss:.4f}  tr={tr_acc:.4f}  val={val_acc:.4f}  lr={lr_now:.6f}')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_weights_v5.pth')
            print(f'  ✓ best: {best_acc:.4f}')

    print(f'\n학습 완료. Best val acc: {best_acc:.4f}')

    # TorchScript export
    model.load_state_dict(torch.load('best_weights_v5.pth', map_location='cpu'))
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
