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
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda:0')
print(f'GPU: {torch.cuda.get_device_name(0)}')

CFG = dict(
    data_root    = './student_data',
    batch_size   = 256,
    num_epochs   = 50,
    lr           = 5e-5,
    weight_decay = 0.01,
    label_smooth = 0.05,
    dropout      = 0.2,
    num_workers  = 4,
    cutmix_alpha = 0.5,
    mixup_alpha  = 0.1,
    checkpoint   = 'best_weights_auto_mobilenet.pth',
    output_pt    = 'my_model_mobilenet_ft.pt',
)


def get_mobilenet(num_classes=200):
    m = mobilenet_v3_large(weights=None, num_classes=num_classes, reduced_tail=True)
    m.features[0][0] = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
    return m


class LabelSmoothingCE(nn.Module):
    def __init__(self, classes=200, smoothing=0.05):
        super().__init__(); self.s=smoothing; self.c=classes
    def forward(self, pred, target):
        lp = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            td = torch.full_like(lp, self.s/(self.c-1))
            td.scatter_(1, target.unsqueeze(1), 1.0-self.s)
        return -(td*lp).sum(-1).mean()


class TinyImageNetVal(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir = os.path.join(root, 'images')
        self.transform = transform
        self.samples = []
        with open(os.path.join(root, 'labels.txt')) as f:
            for line in f:
                if line.strip():
                    fn, cl = line.strip().split('\t')
                    self.samples.append((fn, int(cl)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fn, lb = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fn)).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, lb


def get_loaders(cfg):
    tr = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.1),
    ])
    vl = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tds = datasets.ImageFolder(os.path.join(cfg['data_root'], 'train'), transform=tr)
    vds = TinyImageNetVal(os.path.join(cfg['data_root'], 'public_val'), transform=vl)
    pw = cfg['num_workers'] > 0
    return (DataLoader(tds, batch_size=cfg['batch_size'], shuffle=True,
                       num_workers=cfg['num_workers'], pin_memory=True, persistent_workers=pw),
            DataLoader(vds, batch_size=cfg['batch_size'], shuffle=False,
                       num_workers=cfg['num_workers'], pin_memory=True, persistent_workers=pw))


def cutmix(imgs, labels, alpha=0.5):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    _, _, H, W = imgs.shape
    cx, cy = np.random.randint(W), np.random.randint(H)
    w, h = int(W*math.sqrt(1-lam)), int(H*math.sqrt(1-lam))
    x1, x2 = max(cx-w//2, 0), min(cx+w//2, W)
    y1, y2 = max(cy-h//2, 0), min(cy+h//2, H)
    imgs = imgs.clone()
    imgs[:,:,y1:y2,x1:x2] = imgs[idx,:,y1:y2,x1:x2]
    return imgs, labels, labels[idx], 1-(x2-x1)*(y2-y1)/(W*H)

def mixup(imgs, labels, alpha=0.1):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam*imgs+(1-lam)*imgs[idx], labels, labels[idx], lam


def train_one_epoch(model, loader, optimizer, criterion, scaler, cfg):
    model.train(); tl=cor=tot=0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        r = np.random.rand()
        if r < 0.2: imgs, la, lb, lam = cutmix(imgs, labels, cfg['cutmix_alpha'])
        elif r < 0.35: imgs, la, lb, lam = mixup(imgs, labels, cfg['mixup_alpha'])
        else: la, lb, lam = labels, labels, 1.0
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss = lam*criterion(logits,la) + (1-lam)*criterion(logits,lb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        tl += loss.item()*imgs.size(0)
        cor += (logits.argmax(1)==la).sum().item()
        tot += imgs.size(0)
    return tl/tot, cor/tot


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); cor=tot=0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        cor += (model(imgs).argmax(1)==labels).sum().item()
        tot += imgs.size(0)
    return cor/tot


def main():
    cfg = CFG
    model = get_mobilenet(200).to(DEVICE)
    p = sum(x.numel() for x in model.parameters())
    print(f'Parameters: {p:,}')
    assert p <= 5_000_000

    model.load_state_dict(torch.load(cfg['checkpoint'], map_location=DEVICE, weights_only=True))
    print(f'checkpoint loaded: {cfg["checkpoint"]}')

    train_loader, val_loader = get_loaders(cfg)

    init_val = evaluate(model, val_loader)
    print(f'init val acc: {init_val:.4f}')

    criterion = LabelSmoothingCE(200, cfg['label_smooth'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'], eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = init_val
    for epoch in range(cfg['num_epochs']):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg)
        val_acc = evaluate(model, val_loader)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        print(f'[{epoch+1:3d}/{cfg["num_epochs"]}] loss={tr_loss:.4f} tr={tr_acc:.4f} val={val_acc:.4f} lr={lr_now:.6f}')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_mobilenet_ft.pth')
            print(f'  best: {best_acc:.4f}')

    print(f'\ndone. Best: {best_acc:.4f}')

    model.load_state_dict(torch.load('best_mobilenet_ft.pth', map_location='cpu', weights_only=True))
    model.eval().cpu()
    dummy = torch.randn(1, 3, 64, 64)
    traced = torch.jit.trace(model, dummy)
    torch.jit.save(traced, cfg['output_pt'])
    loaded = torch.jit.load(cfg['output_pt'], map_location='cpu').eval()
    p_count = sum(p.numel() for p in loaded.parameters())
    with torch.no_grad(): out = loaded(dummy)
    assert out.shape == (1, 200)
    assert p_count <= 5_000_000
    print(f'Export OK {p_count:,} params shape {tuple(out.shape)}')

if __name__ == '__main__':
    main()