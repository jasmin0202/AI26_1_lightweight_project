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

DEVICE = torch.device('cuda:0')
print(f'GPU: {torch.cuda.get_device_name(0)}')

CFG = dict(
    data_root    = './student_data',
    batch_size   = 256,
    num_epochs   = 40,
    lr           = 5e-4,
    weight_decay = 0.05,
    label_smooth = 0.1,
    dropout      = 0.3,
    num_workers  = 4,
    cutmix_alpha = 1.0,
    mixup_alpha  = 0.2,
    checkpoint   = 'best_teacher_resnet50.pth',
    output_pth   = 'best_teacher_resnet50_v2.pth',
)


def get_resnet50(num_classes=200, dropout=0.3):
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, num_classes))
    return model


class LabelSmoothingCE(nn.Module):
    def __init__(self, classes=200, smoothing=0.1):
        super().__init__(); self.s=smoothing; self.c=classes
    def forward(self, pred, target):
        lp = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            td = torch.full_like(lp, self.s/(self.c-1))
            td.scatter_(1, target.unsqueeze(1), 1.0-self.s)
        return -(td*lp).sum(-1).mean()


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
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25),
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


def cutmix(imgs, labels, alpha=1.0):
    lam=np.random.beta(alpha,alpha); idx=torch.randperm(imgs.size(0),device=imgs.device)
    _,_,H,W=imgs.shape; cx,cy=np.random.randint(W),np.random.randint(H)
    w,h=int(W*math.sqrt(1-lam)),int(H*math.sqrt(1-lam))
    x1,x2=max(cx-w//2,0),min(cx+w//2,W); y1,y2=max(cy-h//2,0),min(cy+h//2,H)
    imgs=imgs.clone(); imgs[:,:,y1:y2,x1:x2]=imgs[idx,:,y1:y2,x1:x2]
    return imgs,labels,labels[idx],1-(x2-x1)*(y2-y1)/(W*H)

def mixup(imgs, labels, alpha=0.2):
    lam=np.random.beta(alpha,alpha); idx=torch.randperm(imgs.size(0),device=imgs.device)
    return lam*imgs+(1-lam)*imgs[idx],labels,labels[idx],lam


def train_one_epoch(model, loader, optimizer, criterion, scaler, cfg):
    model.train(); tl=cor=tot=0
    for imgs,labels in loader:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
        r=np.random.rand()
        if r<0.4: imgs,la,lb,lam=cutmix(imgs,labels,cfg['cutmix_alpha'])
        elif r<0.7: imgs,la,lb,lam=mixup(imgs,labels,cfg['mixup_alpha'])
        else: la,lb,lam=labels,labels,1.0
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            logits=model(imgs)
            loss=lam*criterion(logits,la)+(1-lam)*criterion(logits,lb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer)
        scaler.update()
        tl+=loss.item()*imgs.size(0); cor+=(logits.argmax(1)==la).sum().item(); tot+=imgs.size(0)
    return tl/tot,cor/tot


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); cor=tot=0
    for imgs,labels in loader:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
        cor+=(model(imgs).argmax(1)==labels).sum().item(); tot+=imgs.size(0)
    return cor/tot


def main():
    cfg = CFG
    model = get_resnet50(200, cfg['dropout']).to(DEVICE)

    model.load_state_dict(torch.load(cfg['checkpoint'], map_location=DEVICE, weights_only=True))
    print(f'checkpoint loaded: {cfg["checkpoint"]}')

    train_loader, val_loader = get_loaders(cfg)

    init_val = evaluate(model, val_loader)
    print(f'init val acc: {init_val:.4f}')

    criterion = LabelSmoothingCE(200, cfg['label_smooth'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['num_epochs'], eta_min=1e-6)
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
            torch.save(model.state_dict(), cfg['output_pth'])
            print(f'  best: {best_acc:.4f}')

    print(f'\ndone. Best: {best_acc:.4f}')


if __name__ == '__main__':
    main()