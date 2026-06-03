import os
import math
import random
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

DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

CFG = dict(
    data_root    = './student_data',
    batch_size   = 256,
    num_epochs   = 300,
    lr           = 2e-3,
    weight_decay = 0.05,
    label_smooth = 0.1,
    dropout      = 0.2,
    num_workers  = 4,
    cutmix_alpha = 1.0,
    mixup_alpha  = 0.2,
    warmup_epoch = 5,
    output_pt    = 'best_weights_resnet18.pt',
)

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
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class TinyResNet18(nn.Module):
    def __init__(self, num_classes=200, width_mult=0.66, dropout=0.2):
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
        self._init_weights()

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_ch, out_ch, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: 
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

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
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(cfg['data_root'], 'train'), transform=train_tf)
    val_ds = TinyImageNetVal(os.path.join(cfg['data_root'], 'public_val'), transform=val_tf)
    print(f'Train: {len(train_ds):,}  Val: {len(val_ds):,}')
    
    pw = cfg['num_workers'] > 0
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True, persistent_workers=pw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=cfg['num_workers'], pin_memory=True, persistent_workers=pw)
    return train_loader, val_loader

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

def train_one_epoch(model, loader, optimizer, criterion, scaler, cfg):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        r = np.random.rand()
        if r < 0.3:
            imgs, la, lb, lam = cutmix(imgs, labels, cfg['cutmix_alpha'])
        elif r < 0.5:
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

def main():
    cfg = CFG

    model = TinyResNet18(num_classes=200, width_mult=0.66, dropout=cfg['dropout']).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}')
    assert params <= 5_000_000, f'Param limit exceeded: {params:,}'
    print('Parameter check passed')

    criterion = LabelSmoothingCE(200, cfg['label_smooth'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    def lr_lambda(epoch):
        warmup = cfg['warmup_epoch']
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / (cfg['num_epochs'] - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda')
    train_loader, val_loader = get_loaders(cfg)

    # CRITICAL: RESUME FROM CHECKPOINT LOGIC
    ckpt_path = 'best_weights_resnet18.pth'
    start_epoch = 0
    best_acc = 0.0

    if os.path.exists(ckpt_path):
        print(f"\n[RESUME SYSTEM] Found existing checkpoint: '{ckpt_path}'")
        # 1. Load the saved weights into model
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        
        # 2. Set start epoch to 277 (the safe point where 0.6428 was achieved)
        start_epoch = 277 
        best_acc = 0.6428  # Lock the historical best score to avoid regression
        
        # 3. FAST-FORWARD Scheduler to align with learning rate of epoch 277
        print(f"[RESUME SYSTEM] Fast-forwarding learning rate scheduler to epoch {start_epoch}...")
        for _ in range(start_epoch):
            scheduler.step()
            
        print(f"[RESUME SYSTEM] Resuming training from Epoch {start_epoch + 1} with LR={optimizer.param_groups[0]['lr']:.6f}\n")
    else:
        print("\n[RESUME SYSTEM] No checkpoint found. Starting training from scratch (Epoch 1).\n")

    for epoch in range(start_epoch, cfg['num_epochs']):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg)
        val_acc = evaluate(model, val_loader)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        print(f'[{epoch+1:3d}/{cfg["num_epochs"]}] loss={tr_loss:.4f}  tr={tr_acc:.4f}  val={val_acc:.4f}  lr={lr_now:.6f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_weights_resnet18.pth')
            print(f'  >> New best acc: {best_acc:.4f}')

    print(f'\nTraining complete. Best public val acc: {best_acc:.4f}')

    # Exporting TorchScript smoothly
    model.load_state_dict(torch.load('best_weights_resnet18.pth', map_location='cpu', weights_only=True))
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
    print(f'Export success: {p_count:,} params, shape {tuple(out.shape)}')

if __name__ == '__main__':
    main()