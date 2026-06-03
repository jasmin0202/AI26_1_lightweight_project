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
import timm  # Ensure to run 'pip install timm' before execution

os.environ["HF_TOKEN"] = ""

import math
import random

# 1. GLOBAL SETTINGS & SEED
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# 2. CONFIGURATION (Optimized for 200 Epochs Time-Attack)
CFG = dict(
    data_root    = './student_data',
    batch_size   = 64,
    num_epochs   = 200,          # Reduced from 300 to 200 for 16h time-attack
    lr           = 1.5e-3,       # Adjusted for 200 epochs
    weight_decay = 0.05,
    label_smooth = 0.1,
    dropout      = 0.2,
    num_workers  = 4,
    cutmix_alpha = 1.0,
    mixup_alpha  = 0.2,
    warmup_epoch = 4,            # Scaled down for 200 epochs
    save_name    = 'best_teacher_convnext.pth', # Separate file name from student!
)

def get_perfect_teacher(num_classes=200):
    teacher = timm.create_model(
        'convnext_tiny', 
        pretrained=True, 
        num_classes=num_classes
    )
    
    teacher.stem[0] = nn.Conv2d(
        in_channels=3, 
        out_channels=teacher.stem[0].out_channels, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=False
    )
    return teacher

# 4. LOSS & DATASET DEFINITION
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
    # Optimized transforms (Removed RandAugment/RandomErasing to save CPU/GPU bottlenecks)
    train_tf = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
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

# 5. DATA AUGMENTATION (CutMix / Mixup)
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

# 6. TRAINING & EVALUATION LOOP
def train_one_epoch(model, loader, optimizer, criterion, scaler, cfg):
    model.train()
    total_loss = correct = total = 0
    
    accumulation_steps = 4  
    optimizer.zero_grad(set_to_none=True)
    
    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        r = np.random.rand()
        if r < 0.3:
            imgs, la, lb, lam = cutmix(imgs, labels, cfg['cutmix_alpha'])
        elif r < 0.5:
            imgs, la, lb, lam = mixup(imgs, labels, cfg['mixup_alpha'])
        else:
            la, lb, lam = labels, labels, 1.0
            
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss = (lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)) / accumulation_steps
            
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += (loss.item() * accumulation_steps) * imgs.size(0)
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

# 7. MAIN FUNCTION
def main():
    cfg = CFG

    # Model instantiation (ConvNeXt-Tiny)
    model = get_perfect_teacher(num_classes=200).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f'Teacher Parameters: {params:,}')
    print('No parameter constraint for the teacher model. Ready to train.')

    criterion = LabelSmoothingCE(200, cfg['label_smooth'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    # Cosine Annealing Lambda adjusted for 200 epochs
    def lr_lambda(epoch):
        warmup = cfg['warmup_epoch']
        if epoch < warmup:
            return (epoch + 1) / warmup
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
            torch.save(model.state_dict(), cfg['save_name'])
            print(f'  >> New Best Teacher Acc: {best_acc:.4f}')

    print(f'\nTeacher Training Complete. Best public val acc: {best_acc:.4f}')
    print(f"Weight file successfully saved to '{cfg['save_name']}'")

if __name__ == '__main__':
    main()