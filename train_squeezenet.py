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

DEVICE = torch.device('cuda:3')
print(f'GPU: {torch.cuda.get_device_name(3)}')

CFG = dict(
    data_root    = './student_data',
    batch_size   = 256,
    num_epochs   = 200,
    lr           = 4e-3,
    weight_decay = 0.01,
    label_smooth = 0.1,
    dropout      = 0.2,
    num_workers  = 4,
    cutmix_alpha = 0.8,
    mixup_alpha  = 0.2,
    save_epochs  = [150, 170, 190, 200],
    output_pt    = 'my_model_squeezenet.pt',
)


class FireModule(nn.Module):
    def __init__(self, in_ch, s1, e1, e3):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_ch, s1, 1), nn.ReLU(inplace=True))
        self.expand1 = nn.Sequential(
            nn.Conv2d(s1, e1, 1), nn.ReLU(inplace=True))
        self.expand3 = nn.Sequential(
            nn.Conv2d(s1, e3, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([self.expand1(x), self.expand3(x)], dim=1)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=200, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 3, 1, 1), nn.ReLU(inplace=True),
            FireModule(96,  24, 96,  96),
            FireModule(192, 24, 96,  96),
            nn.MaxPool2d(2, 2),
            FireModule(192, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            nn.MaxPool2d(2, 2),
            FireModule(384, 80, 320, 320),
            FireModule(640, 80, 320, 320),
            FireModule(640, 96, 400, 400),
            FireModule(800, 96, 400, 400),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(800, num_classes, 1), 
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.head(self.features(x))


class LabelSmoothingCE(nn.Module):
    def __init__(self,classes=200,smoothing=0.1):
        super().__init__(); self.s=smoothing; self.c=classes
    def forward(self,pred,target):
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
        transforms.RandomCrop(64,padding=6),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3,0.3,0.3,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.15),
    ])
    vl=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tds=datasets.ImageFolder(os.path.join(cfg['data_root'],'train'),transform=tr)
    vds=TinyImageNetVal(os.path.join(cfg['data_root'],'public_val'),transform=vl)
    pw=cfg['num_workers']>0
    return (DataLoader(tds,batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['num_workers'],pin_memory=True,persistent_workers=pw),
            DataLoader(vds,batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['num_workers'],pin_memory=True,persistent_workers=pw))


def cutmix(imgs,labels,alpha=0.8):
    lam=np.random.beta(alpha,alpha); idx=torch.randperm(imgs.size(0),device=imgs.device)
    _,_,H,W=imgs.shape; cx,cy=np.random.randint(W),np.random.randint(H)
    w,h=int(W*math.sqrt(1-lam)),int(H*math.sqrt(1-lam))
    x1,x2=max(cx-w//2,0),min(cx+w//2,W); y1,y2=max(cy-h//2,0),min(cy+h//2,H)
    imgs=imgs.clone(); imgs[:,:,y1:y2,x1:x2]=imgs[idx,:,y1:y2,x1:x2]
    return imgs,labels,labels[idx],1-(x2-x1)*(y2-y1)/(W*H)

def mixup(imgs,labels,alpha=0.2):
    lam=np.random.beta(alpha,alpha); idx=torch.randperm(imgs.size(0),device=imgs.device)
    return lam*imgs+(1-lam)*imgs[idx],labels,labels[idx],lam


def train_one_epoch(model,loader,optimizer,criterion,scaler,scheduler,cfg):
    model.train(); tl=cor=tot=0
    for imgs,labels in loader:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
        r=np.random.rand()
        if r<0.3: imgs,la,lb,lam=cutmix(imgs,labels,cfg['cutmix_alpha'])
        elif r<0.5: imgs,la,lb,lam=mixup(imgs,labels,cfg['mixup_alpha'])
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
        scheduler.step()
        tl+=loss.item()*imgs.size(0); cor+=(logits.argmax(1)==la).sum().item(); tot+=imgs.size(0)
    return tl/tot,cor/tot


@torch.no_grad()
def evaluate(model,loader):
    model.eval(); cor=tot=0
    for imgs,labels in loader:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
        cor+=(model(imgs).argmax(1)==labels).sum().item(); tot+=imgs.size(0)
    return cor/tot


def main():
    cfg=CFG
    model=SqueezeNet(200,cfg['dropout']).to(DEVICE)
    p=sum(x.numel() for x in model.parameters())
    print(f'Parameters: {p:,}')
    assert p<=5_000_000, f'params exceed: {p:,}'
    print('params OK')

    train_loader,val_loader=get_loaders(cfg)
    criterion=LabelSmoothingCE(200,cfg['label_smooth'])
    optimizer=optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=optim.lr_scheduler.OneCycleLR(
        optimizer,max_lr=cfg['lr'],epochs=cfg['num_epochs'],
        steps_per_epoch=len(train_loader),pct_start=0.1,
        anneal_strategy='cos',div_factor=25.0,final_div_factor=1e4,
    )
    scaler=torch.amp.GradScaler('cuda')

    best_acc=0.0
    for epoch in range(cfg['num_epochs']):
        tr_loss,tr_acc=train_one_epoch(model,train_loader,optimizer,criterion,scaler,scheduler,cfg)
        val_acc=evaluate(model,val_loader)
        lr_now=optimizer.param_groups[0]['lr']
        if (epoch+1)%10==0 or epoch==0:
            print(f'[{epoch+1:3d}/{cfg["num_epochs"]}] loss={tr_loss:.4f} tr={tr_acc:.4f} val={val_acc:.4f} lr={lr_now:.5f}')
        if val_acc>best_acc:
            best_acc=val_acc
            torch.save(model.state_dict(),'best_squeezenet.pth')
            print(f'  best: {best_acc:.4f}')
        if (epoch+1) in cfg['save_epochs']:
            path=f'ckpt_squeezenet_ep{epoch+1}.pth'
            torch.save(model.state_dict(),path)
            print(f'  checkpoint: {path} (val={val_acc:.4f})')

    print(f'\ndone. Best: {best_acc:.4f}')

    model.load_state_dict(torch.load('best_squeezenet.pth',map_location='cpu',weights_only=True))
    model.eval().cpu()
    dummy=torch.randn(1,3,64,64)
    traced=torch.jit.trace(model,dummy)
    torch.jit.save(traced,cfg['output_pt'])
    loaded=torch.jit.load(cfg['output_pt'],map_location='cpu').eval()
    p_count=sum(p.numel() for p in loaded.parameters())
    with torch.no_grad(): out=loaded(dummy)
    assert out.shape==(1,200)
    assert p_count<=5_000_000
    print(f'Export OK {p_count:,} params shape {tuple(out.shape)}')

if __name__=='__main__':
    main()