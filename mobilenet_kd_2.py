import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda:1')
print(f'GPU: {torch.cuda.get_device_name(3)}')

CFG = dict(
    data_root       = './student_data',
    batch_size      = 256,
    num_epochs      = 25,            
    lr              = 1e-5,          
    weight_decay    = 0.02,
    label_smooth    = 0.1,
    dropout         = 0.3,
    num_workers     = 4,
    kd_temperature  = 4.0,          
    kd_alpha        = 1.0,          
    student_ckpt    = 'best_weights_auto_mobilenet.pth',
    teacher_ckpt    = 'best_teacher_resnet50_v2.pth',
    output_pth      = 'best_mobilenet_kd_pure.pth',
    output_pt       = 'my_model_mobilenet_kd_pure.pt',
)

def get_resnet50(num_classes=200, dropout=0.3):
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, num_classes))
    return model

def get_mobilenet(num_classes=200):
    m = models.mobilenet_v3_large(weights=None, num_classes=num_classes, reduced_tail=True)
    m.features[0][0] = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
    return m

def kd_loss(student_logits, teacher_logits, temperature):
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction='batchmean') * (temperature ** 2)

class TinyImageNetVal(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir=os.path.join(root,'images'); self.transform=transform; self.samples=[]
        with open(os.path.join(root,'labels.txt')) as f:
            for line in f:
                if line.strip():
                    fn,cl=line.strip().split('\t'); self.samples.append((fn,int(cl)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fn,lb=self.samples[idx]
        img=Image.open(os.path.join(self.img_dir,fn)).convert('RGB')
        if self.transform: img=self.transform(img)
        return img,lb

def get_loaders(cfg):
    tr = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    vl = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tds = datasets.ImageFolder(os.path.join(cfg['data_root'],'train'), transform=tr)
    vds = TinyImageNetVal(os.path.join(cfg['data_root'],'public_val'), transform=vl)
    pw = cfg['num_workers'] > 0
    return (DataLoader(tds, batch_size=cfg['batch_size'], shuffle=True,
                       num_workers=cfg['num_workers'], pin_memory=True, persistent_workers=pw),
            DataLoader(vds, batch_size=cfg['batch_size'], shuffle=False,
                       num_workers=cfg['num_workers'], pin_memory=True, persistent_workers=pw))

def train_one_epoch(student, teacher, loader, optimizer, scaler, cfg):
    student.train(); teacher.eval()
    tl=cor=tot=0
    T     = cfg['kd_temperature']

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            s_logits = student(imgs)
            with torch.no_grad():
                t_logits = teacher(imgs)
            loss = kd_loss(s_logits, t_logits, T)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        tl  += loss.item() * imgs.size(0)
        cor += (s_logits.argmax(1) == labels).sum().item()
        tot += imgs.size(0)
    return tl/tot, cor/tot

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); cor=tot=0
    for imgs,labels in loader:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
        cor+=(model(imgs).argmax(1)==labels).sum().item(); tot+=imgs.size(0)
    return cor/tot

def main():
    cfg = CFG
    teacher = get_resnet50(200, dropout=0.3).to(DEVICE)
    teacher.load_state_dict(torch.load(cfg['teacher_ckpt'], map_location=DEVICE, weights_only=True))
    teacher.eval()
    print(f'Teacher loaded: {cfg["teacher_ckpt"]}')

    student = get_mobilenet(200).to(DEVICE)
    student.load_state_dict(torch.load(cfg['student_ckpt'], map_location=DEVICE, weights_only=True))
    print(f'Student loaded: {cfg["student_ckpt"]}')

    train_loader, val_loader = get_loaders(cfg)

    t_acc = evaluate(teacher, val_loader)
    s_acc = evaluate(student, val_loader)
    print(f'Teacher val acc: {t_acc:.4f}')
    print(f'Student val acc: {s_acc:.4f}')

    optimizer = optim.AdamW(student.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'], eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = s_acc
    for epoch in range(cfg['num_epochs']):
        tr_loss, tr_acc = train_one_epoch(student, teacher, train_loader, optimizer, scaler, cfg)
        val_acc = evaluate(student, val_loader)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        print(f'[{epoch+1:3d}/{cfg["num_epochs"]}] loss={tr_loss:.4f} tr={tr_acc:.4f} val={val_acc:.4f} lr={lr_now:.6f}')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), cfg['output_pth'])
            print(f'  best: {best_acc:.4f}')

    print(f'\ndone. Best: {best_acc:.4f}')
    student.load_state_dict(torch.load(cfg['output_pth'], map_location='cpu', weights_only=True))
    student.eval().cpu()
    dummy = torch.randn(1, 3, 64, 64)
    traced = torch.jit.trace(student, dummy)
    torch.jit.save(traced, cfg['output_pt'])
    print(f'Export OK')

if __name__ == '__main__':
    main()