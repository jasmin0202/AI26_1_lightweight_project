
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
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda:0')
print(f'GPU: {torch.cuda.get_device_name(0)}')

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
    aux_weight   = 0.3,   
    save_epochs  = [200, 230, 260, 280, 300],
    output_pt    = 'my_model_sam_resnet_mlisd.pt',
)


# Model (MLISD)
class BasicBlock(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.c1=nn.Conv2d(ic,oc,3,stride,1,bias=False); self.b1=nn.BatchNorm2d(oc)
        self.c2=nn.Conv2d(oc,oc,3,1,1,bias=False); self.b2=nn.BatchNorm2d(oc)
        self.relu=nn.ReLU(inplace=True)
        self.skip=nn.Sequential()
        if stride!=1 or ic!=oc:
            self.skip=nn.Sequential(nn.Conv2d(ic,oc,1,stride,bias=False),nn.BatchNorm2d(oc))
    def forward(self,x):
        return self.relu(self.b2(self.c2(self.relu(self.b1(self.c1(x)))))+self.skip(x))


class AuxClassifier(nn.Module):
    def __init__(self, in_ch, num_classes=200, dropout=0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )
    def forward(self, x):
        return self.fc(self.pool(x))


class TinyResNet18MLISD(nn.Module):
    def __init__(self, num_classes=200, width_mult=0.48, dropout=0.2):
        super().__init__()
        c = [int(64*width_mult), int(128*width_mult),
             int(256*width_mult), int(512*width_mult)]

        self.stem   = nn.Sequential(
            nn.Conv2d(3,c[0],3,1,1,bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(BasicBlock(c[0],c[0]), BasicBlock(c[0],c[0]))
        self.layer2 = nn.Sequential(BasicBlock(c[0],c[1],2), BasicBlock(c[1],c[1]))
        self.layer3 = nn.Sequential(BasicBlock(c[1],c[2],2), BasicBlock(c[2],c[2]))
        self.layer4 = nn.Sequential(BasicBlock(c[2],c[3],2), BasicBlock(c[3],c[3]))
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Sequential(nn.Dropout(dropout), nn.Linear(c[3], num_classes))

        self.aux2 = AuxClassifier(c[1], num_classes, dropout)
        self.aux3 = AuxClassifier(c[2], num_classes, dropout)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x2 = self.layer2(x);  aux2_out = self.aux2(x2)
        x3 = self.layer3(x2); aux3_out = self.aux3(x3)
        x4 = self.layer4(x3)
        out = self.fc(torch.flatten(self.gap(x4), 1))
        if self.training:
            return out, aux2_out, aux3_out
        return out


# Loss / Data
class LabelSmoothingCE(nn.Module):
    def __init__(self, classes=200, smoothing=0.1):
        super().__init__(); self.s=smoothing; self.c=classes
    def forward(self, pred, target):
        lp=F.log_softmax(pred,dim=-1)
        with torch.no_grad():
            td=torch.full_like(lp,self.s/(self.c-1))
            td.scatter_(1,target.unsqueeze(1),1.0-self.s)
        return -(td*lp).sum(-1).mean()


class TinyImageNetVal(Dataset):
    def __init__(self,root,transform=None):
        self.img_dir=os.path.join(root,'images'); self.transform=transform; self.samples=[]
        with open(os.path.join(root,'labels.txt')) as f:
            for line in f:
                if line.strip():
                    fn,cl=line.strip().split('\t'); self.samples.append((fn,int(cl)))
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        fn,lb=self.samples[idx]
        img=Image.open(os.path.join(self.img_dir,fn)).convert('RGB')
        if self.transform: img=self.transform(img)
        return img,lb


def get_loaders(cfg):
    tr=transforms.Compose([
        transforms.RandomCrop(64,padding=8), transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2,magnitude=7),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25),
    ])
    vl=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tds=datasets.ImageFolder(os.path.join(cfg['data_root'],'train'),transform=tr)
    vds=TinyImageNetVal(os.path.join(cfg['data_root'],'public_val'),transform=vl)
    pw=cfg['num_workers']>0
    return (DataLoader(tds,batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['num_workers'],pin_memory=True,persistent_workers=pw),
            DataLoader(vds,batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['num_workers'],pin_memory=True,persistent_workers=pw))


def cutmix(imgs,labels,alpha=1.0):
    lam=np.random.beta(alpha,alpha); idx=torch.randperm(imgs.size(0),device=imgs.device)
    _,_,H,W=imgs.shape; cx,cy=np.random.randint(W),np.random.randint(H)
    w,h=int(W*math.sqrt(1-lam)),int(H*math.sqrt(1-lam))
    x1,x2=max(cx-w//2,0),min(cx+w//2,W); y1,y2=max(cy-h//2,0),min(cy+h//2,H)
    imgs=imgs.clone(); imgs[:,:,y1:y2,x1:x2]=imgs[idx,:,y1:y2,x1:x2]
    return imgs,labels,labels[idx],1-(x2-x1)*(y2-y1)/(W*H)

def mixup(imgs,labels,alpha=0.2):
    lam=np.random.beta(alpha,alpha); idx=torch.randperm(imgs.size(0),device=imgs.device)
    return lam*imgs+(1-lam)*imgs[idx],labels,labels[idx],lam


# Train / Eval

def train_one_epoch(model, loader, optimizer, criterion, scaler, cfg):
    model.train(); tl=cor=tot=0
    aw = cfg['aux_weight']
    for imgs,labels in loader:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
        r=np.random.rand()
        if r<0.25: imgs,la,lb,lam=cutmix(imgs,labels,cfg['cutmix_alpha'])
        elif r<0.45: imgs,la,lb,lam=mixup(imgs,labels,cfg['mixup_alpha'])
        else: la,lb,lam=labels,labels,1.0
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out,aux2,aux3=model(imgs)
            main_loss=lam*criterion(out,la)+(1-lam)*criterion(out,lb)
            a2_loss=lam*criterion(aux2,la)+(1-lam)*criterion(aux2,lb)
            a3_loss=lam*criterion(aux3,la)+(1-lam)*criterion(aux3,lb)
            loss=main_loss+aw*(a2_loss+a3_loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer)
        scaler.update()
        tl+=loss.item()*imgs.size(0)
        cor+=(out.argmax(1)==la).sum().item()
        tot+=imgs.size(0)
    return tl/tot,cor/tot


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); cor=tot=0
    for imgs,labels in loader:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
        out = model(imgs)   
        cor+=(out.argmax(1)==labels).sum().item(); tot+=imgs.size(0)
    return cor/tot


# Main
def main():
    cfg=CFG
    model=TinyResNet18MLISD(200, 0.48, cfg['dropout']).to(DEVICE)
    p=sum(x.numel() for x in model.parameters())
    print(f'Parameters: {p:,}')
    assert p<=2_900_000, f'params exceed: {p:,}'
    print('params OK')

    criterion=LabelSmoothingCE(200,cfg['label_smooth'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )
    def lr_lambda(e):
        w=cfg['warmup_epoch']
        if e<w: return (e+1)/w
        return 0.5*(1+math.cos(math.pi*(e-w)/(cfg['num_epochs']-w)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda')
    train_loader,val_loader=get_loaders(cfg)

    best_acc=0.0
    for epoch in range(cfg['num_epochs']):
        tr_loss,tr_acc=train_one_epoch(model,train_loader,optimizer,criterion,scaler,cfg)
        val_acc=evaluate(model,val_loader)
        scheduler.step()
        lr_now=optimizer.param_groups[0]['lr']
        if (epoch+1)%10==0 or epoch==0:
            print(f'[{epoch+1:3d}/{cfg["num_epochs"]}] loss={tr_loss:.4f} tr={tr_acc:.4f} val={val_acc:.4f} lr={lr_now:.5f}')
        if val_acc>best_acc:
            best_acc=val_acc
            torch.save(model.state_dict(),'best_sam_resnet_mlisd.pth')
            print(f'  ? best: {best_acc:.4f}')
        if (epoch+1) in cfg['save_epochs']:
            path=f'ckpt_resnet_mlisd_ep{epoch+1}.pth'
            torch.save(model.state_dict(), path)
            print(f'  >> checkpoint saved: {path} (val={val_acc:.4f})')

    print(f'\n done. Best: {best_acc:.4f}')

    model.load_state_dict(torch.load('best_sam_resnet_mlisd.pth',map_location='cpu',weights_only=True))
    model.eval().cpu()
    dummy=torch.randn(1,3,64,64)
    traced=torch.jit.trace(model,dummy)
    torch.jit.save(traced,cfg['output_pt'])
    loaded=torch.jit.load(cfg['output_pt'],map_location='cpu').eval()
    p_count=sum(p.numel() for p in loaded.parameters())
    with torch.no_grad(): out=loaded(dummy)
    assert out.shape==(1,200)
    assert p_count<=5_000_000
    print(f'? Export OK ? {p_count:,} params, shape {tuple(out.shape)}')

if __name__=='__main__':
    main()