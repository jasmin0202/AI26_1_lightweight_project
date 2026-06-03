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
from torchvision.models import mobilenet_v3_large
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

CFG = dict(
    data_root    = './student_data',
    batch_size   = 256,
    num_epochs   = 15,          # Fast SAM fine-tuning epochs
    lr           = 2e-5,          # Low learning rate for preserving weights
    weight_decay = 0.05,
    label_smooth = 0.1,
    dropout      = 0.2,
    num_workers  = 4,
    cutmix_alpha = 1.0,
    mixup_alpha  = 0.2,
    warmup_epoch = 0,           # No warmup needed for fine-tuning
    output_pt    = 'my_model_resnet18.pt',
    use_sam      = True,        # Toggle Switch for SAM Tuning
    load_existing = True        # Load your existing 0.65 checkpoints
)

class SAM(torch.optim.Optimizer):
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
                e_w = p.grad * scale.to(p)
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

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]), p=2
               )
        return norm

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
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch4, num_classes))
    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks): layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.fc(torch.flatten(self.gap(self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))), 1))

def get_mobilenet_v3(num_classes=200):
    model = mobilenet_v3_large(num_classes=num_classes, reduced_tail=True)
    model.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    return model

class EnsembleWrapper(nn.Module):
    def __init__(self, modelA, modelB):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
    def forward(self, x):
        return (self.modelA(x) + self.modelB(x)) / 2.0

class LabelSmoothingCE(nn.Module):
    def __init__(self, classes=200, smoothing=0.1):
        super().__init__()
        self.smoothing, self.cls = smoothing, classes
    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(true_dist * log_probs).sum(dim=-1).mean()

class TinyImageNetVal(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir, self.transform, self.samples = os.path.join(root, 'images'), transform, []
        with open(os.path.join(root, 'labels.txt')) as f:
            for line in f:
                if line.strip():
                    fname, cls = line.strip().split('\t')
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
    return DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True, persistent_workers=True), DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True, persistent_workers=True)

def cutmix(imgs, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    _, _, H, W = imgs.shape
    cx, cy = np.random.randint(W), np.random.randint(H)
    w, h = int(W * math.sqrt(1 - lam)), int(H * math.sqrt(1 - lam))
    x1, x2 = max(cx - w//2, 0), min(cx + w//2, W)
    y1, y2 = max(cy - h//2, 0), min(cy + h//2, H)
    imgs = imgs.clone()
    imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    return imgs, labels, labels[idx], 1 - (x2-x1)*(y2-y1)/(W*H)

def mixup(imgs, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1 - lam) * imgs[idx], labels, labels[idx], lam

def train_one_epoch(model, loader, optimizer, criterion, scaler, cfg):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        r = np.random.rand()
        if r < 0.3: imgs, la, lb, lam = cutmix(imgs, labels, cfg['cutmix_alpha'])
        elif r < 0.5: imgs, la, lb, lam = mixup(imgs, labels, cfg['mixup_alpha'])
        else: la, lb, lam = labels, labels, 1.0
        
        if cfg['use_sam']:
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            with torch.amp.autocast('cuda'):
                logits_second = model(imgs)
                loss_second = lam * criterion(logits_second, la) + (1 - lam) * criterion(logits_second, lb)
            loss_second.backward()
            optimizer.second_step(zero_grad=True)
            
            logits = logits_second
            loss = loss_second
        else:
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
        correct += (logits.argmax(1) == la).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return correct / total

def run_training(model_name, model, train_loader, val_loader, cfg):
    print(f'\n--- Start Training: {model_name.upper()} ---')
    criterion = LabelSmoothingCE(200, cfg['label_smooth'])
    
    if cfg['use_sam']:
        base_optimizer = optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        
    if cfg['warmup_epoch'] > 0:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: (e + 1) / cfg['warmup_epoch'] if e < cfg['warmup_epoch'] else 0.5 * (1 + math.cos(math.pi * (e - cfg['warmup_epoch']) / (cfg['num_epochs'] - cfg['warmup_epoch']))) )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'], eta_min=1e-6)
        
    scaler = torch.amp.GradScaler('cuda')
    best_acc = 0.0
    weight_path = f'best_weights_auto_{model_name}.pth'
    
    if cfg['load_existing'] and os.path.exists(f'best_weights_auto_{model_name}.pth'):
        print(f'>> Loading existing weights for {model_name.upper()} to fine-tune...')
        model.load_state_dict(torch.load(f'best_weights_auto_{model_name}.pth', map_location=DEVICE))
        best_acc = evaluate(model, val_loader)
        print(f'>> Baseline Val Acc: {best_acc:.4f}')
    
    for epoch in range(cfg['num_epochs']):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg)
        val_acc = evaluate(model, val_loader)
        if cfg['use_sam']:
            optimizer._step_count = 1
            
        scheduler.step() 
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'[{model_name.upper()}] Epoch {epoch+1:3d}: loss={tr_loss:.4f} tr={tr_acc:.4f} val={val_acc:.4f}')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), weight_path)
            print(f'  >> [{model_name.upper()}] New Best: {best_acc:.4f}')
            
    print(f'--- Finished {model_name.upper()} | Best Val: {best_acc:.4f} ---\n')
    return weight_path

def main():
    cfg = CFG
    m1_check = TinyResNet18(num_classes=200, width_mult=0.48, dropout=cfg['dropout'])
    m2_check = get_mobilenet_v3(num_classes=200)
    
    p1 = sum(p.numel() for p in m1_check.parameters())
    p2 = sum(p.numel() for p in m2_check.parameters())
    total_p = p1 + p2
    
    print("\n================================")
    print(" [UPFRONT PARAMETER VALIDATION]")
    print(f"  - ResNet18 Member Params : {p1:,}")
    print(f"  - MobileNetV3 Member Params : {p2:,}")
    print(f"  - Combined Ensemble Params  : {total_p:,}")
    print("================================")
    
    assert total_p <= 5_000_000, f"CRITICAL ERROR: Parameter limit exceeded! Total: {total_p:,} > 5,000,000"
    print(">> Parameter check PASSED. Safe to proceed.\n")
    del m1_check, m2_check
    
    train_loader, val_loader = get_loaders(cfg)

    m1 = TinyResNet18(num_classes=200, width_mult=0.48, dropout=cfg['dropout']).to(DEVICE)
    w1_path = run_training('resnet', m1, train_loader, val_loader, cfg)

    m2 = get_mobilenet_v3(num_classes=200).to(DEVICE)
    w2_path = run_training('mobilenet', m2, train_loader, val_loader, cfg)

    print('\n--- Starting Auto-Ensemble Export ---')
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    m1.load_state_dict(torch.load(w1_path, map_location='cpu'))
    m2.load_state_dict(torch.load(w2_path, map_location='cpu'))
    
    ensemble_model = EnsembleWrapper(m1.cpu(), m2.cpu()).eval()
    
    dummy = torch.randn(1, 3, 64, 64)
    traced = torch.jit.trace(ensemble_model, dummy)
    torch.jit.save(traced, cfg['output_pt'])
    print(f'>> Successfully exported single file: {cfg["output_pt"]}')

if __name__ == '__main__':
    main()