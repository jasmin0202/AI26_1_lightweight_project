"""
SAM-based Ensemble Training
- TinyResNet18 + MobileNetV3
- SAM optimizer (Sharpness Aware Minimization)
- padding=8, lr=4e-3, warmup=10, magnitude=7
- 300 epoch scratch
- ResNet: cuda:0 / MobileNet: cuda:3
"""

import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v3_large
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

CFG = dict(
    data_root    = './student_data',
    batch_size   = 256,
    num_epochs   = 300,
    lr           = 4e-3,
    weight_decay = 0.01,
    label_smooth = 0.1,
    dropout      = 0.2,
    num_workers  = 4,
    cutmix_alpha = 1.0,
    mixup_alpha  = 0.2,
    warmup_epoch = 10,
    rho          = 0.05,   # SAM perturbation radius
)


# 1. SAM Optimizer
class SAM(optim.Optimizer):
    """Sharpness Aware Minimization (Foret et al., 2021)"""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        return torch.norm(torch.stack([
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group['params']
            if p.grad is not None
        ]), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# 2. Models
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class TinyResNet18(nn.Module):
    def __init__(self, num_classes=200, width_mult=0.48, dropout=0.2):
        super().__init__()
        c = [int(64*width_mult), int(128*width_mult),
             int(256*width_mult), int(512*width_mult)]
        self.stem   = nn.Sequential(
            nn.Conv2d(3, c[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True)
        )
        self.layer1 = self._make(c[0], c[0], 2, 1)
        self.layer2 = self._make(c[0], c[1], 2, 2)
        self.layer3 = self._make(c[1], c[2], 2, 2)
        self.layer4 = self._make(c[2], c[3], 2, 2)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Sequential(nn.Dropout(dropout), nn.Linear(c[3], num_classes))
    def _make(self, ic, oc, n, s):
        return nn.Sequential(BasicBlock(ic, oc, s),
                             *[BasicBlock(oc, oc) for _ in range(n-1)])
    def forward(self, x):
        return self.fc(torch.flatten(self.gap(
            self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))), 1))


def get_mobilenet(num_classes=200):
    m = mobilenet_v3_large(weights=None, num_classes=num_classes, reduced_tail=True)
    m.features[0][0] = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
    return m


class EnsembleWrapper(nn.Module):
    def __init__(self, a: nn.Module, b: nn.Module):
        super().__init__()
        self.a = a
        self.b = b
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a(x) + self.b(x)) * 0.5

# 3. Loss / Data
class LabelSmoothingCE(nn.Module):
    def __init__(self, classes=200, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.cls = classes
    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(true_dist * log_probs).sum(dim=-1).mean()


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
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25),
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
    return (
        DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                   num_workers=cfg['num_workers'], pin_memory=True, persistent_workers=pw),
        DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                   num_workers=cfg['num_workers'], pin_memory=True, persistent_workers=pw)
    )


# 4. Augmentation
def cutmix(imgs, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    _, _, H, W = imgs.shape
    cx, cy = np.random.randint(W), np.random.randint(H)
    w = int(W * math.sqrt(1 - lam))
    h = int(H * math.sqrt(1 - lam))
    x1, x2 = max(cx-w//2, 0), min(cx+w//2, W)
    y1, y2 = max(cy-h//2, 0), min(cy+h//2, H)
    imgs = imgs.clone()
    imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    return imgs, labels, labels[idx], 1-(x2-x1)*(y2-y1)/(W*H)

def mixup(imgs, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam*imgs+(1-lam)*imgs[idx], labels, labels[idx], lam


# 5. Train / Eval 
def train_one_epoch_sam(model, loader, optimizer, criterion, cfg, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        r = np.random.rand()
        if r < 0.25:
            imgs, la, lb, lam = cutmix(imgs, labels, cfg['cutmix_alpha'])
        elif r < 0.45:
            imgs, la, lb, lam = mixup(imgs, labels, cfg['mixup_alpha'])
        else:
            la, lb, lam = labels, labels, 1.0

        # SAM first step
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss = lam * criterion(logits, la) + (1-lam) * criterion(logits, lb)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # SAM second step
        with torch.amp.autocast('cuda'):
            logits2 = model(imgs)
            loss2 = lam * criterion(logits2, la) + (1-lam) * criterion(logits2, lb)
        loss2.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.second_step(zero_grad=True)

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == la).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total   += imgs.size(0)
    return correct / total

# 6. Training runner

def run_training(name, model, train_loader, val_loader, cfg, device, save_path):
    print(f'\n{"="*50}')
    print(f' Training: {name}  |  device: {device}')
    print(f'{"="*50}')

    criterion = LabelSmoothingCE(200, cfg['label_smooth'])
    base_opt  = optim.AdamW
    optimizer = SAM(model.parameters(), base_opt,
                    rho=cfg['rho'], lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    def lr_lambda(epoch):
        warmup = cfg['warmup_epoch']
        if epoch < warmup: return (epoch + 1) / warmup
        progress = (epoch - warmup) / (cfg['num_epochs'] - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer.base_optimizer, lr_lambda)

    best_acc = 0.0
    for epoch in range(cfg['num_epochs']):
        tr_loss, tr_acc = train_one_epoch_sam(model, train_loader, optimizer, criterion, cfg, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        lr_now = optimizer.base_optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'[{name}][{epoch+1:3d}/{cfg["num_epochs"]}] '
                  f'loss={tr_loss:.4f} tr={tr_acc:.4f} val={val_acc:.4f} lr={lr_now:.5f}')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'  ? [{name}] best: {best_acc:.4f}')

    print(f'\n[{name}] done. Best val acc: {best_acc:.4f}')
    return best_acc


# 7. Main
def main():
    cfg = CFG

    m1_check = TinyResNet18(num_classes=200, width_mult=0.48, dropout=cfg['dropout'])
    m2_check = get_mobilenet(num_classes=200)
    p1 = sum(p.numel() for p in m1_check.parameters())
    p2 = sum(p.numel() for p in m2_check.parameters())
    print(f'\n{"="*40}')
    print(f' ResNet params    : {p1:,}')
    print(f' MobileNet params : {p2:,}')
    print(f' sum            : {p1+p2:,}')
    print(f'{"="*40}')
    assert p1 + p2 <= 5_000_000, f'param exceed: {p1+p2:,}'
    print(' param good\n')
    del m1_check, m2_check

    train_loader, val_loader = get_loaders(cfg)

    dev1 = torch.device('cuda:0')
    m1   = TinyResNet18(num_classes=200, width_mult=0.48, dropout=cfg['dropout']).to(dev1)
    run_training('ResNet', m1, train_loader, val_loader, cfg, dev1, 'best_sam_resnet.pth')

    dev2 = torch.device('cuda:3')
    m2   = get_mobilenet(num_classes=200).to(dev2)
    run_training('MobileNet', m2, train_loader, val_loader, cfg, dev2, 'best_sam_mobilenet.pth')
    print('\n ensemble export...')
    m1.load_state_dict(torch.load('best_sam_resnet.pth',    map_location='cpu', weights_only=True))
    m2.load_state_dict(torch.load('best_sam_mobilenet.pth', map_location='cpu', weights_only=True))
    m1.cpu().eval()
    m2.cpu().eval()

    ensemble = EnsembleWrapper(m1, m2).eval()
    dummy    = torch.randn(1, 3, 64, 64)
    traced   = torch.jit.trace(ensemble, dummy)
    torch.jit.save(traced, 'my_model_sam_ensemble.pt')

    loaded  = torch.jit.load('my_model_sam_ensemble.pt', map_location='cpu').eval()
    p_total = sum(p.numel() for p in loaded.parameters())
    with torch.no_grad():
        out = loaded(dummy)
    assert out.shape == (1, 200)
    assert p_total <= 5_000_000, f'param exceed: {p_total:,}'
    print(f'Export OK  {p_total:,} params, shape {tuple(out.shape)}')
    print('export ok: my_model_sam_ensemble.pt')


if __name__ == '__main__':
    main()