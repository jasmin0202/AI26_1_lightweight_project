"""
SAM ResNet ? cuda:0
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
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda:3')
print(f'GPU: {torch.cuda.get_device_name(3)}')

CFG = dict(
    data_root    = './student_data',
    batch_size   = 256,
    num_epochs   = 300,
    lr           = 2e-3,
    weight_decay = 0.01,
    label_smooth = 0.1,
    dropout      = 0.2,
    num_workers  = 4,
    cutmix_alpha = 1.0,
    mixup_alpha  = 0.2,
    warmup_epoch = 10,
    rho          = 0.05,
    output_pt    = 'my_model_sam_resnet.pt',
)

class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        super().__init__(params, dict(rho=rho, **kwargs))
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = torch.norm(torch.stack([
            p.grad.norm(p=2) for g in self.param_groups
            for p in g['params'] if p.grad is not None
        ]), p=2)
        for g in self.param_groups:
            scale = g['rho'] / (grad_norm + 1e-12)
            for p in g['params']:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w); self.state[p]['e_w'] = e_w
        if zero_grad: self.zero_grad()
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

class BasicBlock(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(ic, oc, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(oc)
        self.c2 = nn.Conv2d(oc, oc, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(oc)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Sequential()
        if stride != 1 or ic != oc:
            self.skip = nn.Sequential(nn.Conv2d(ic, oc, 1, stride, bias=False), nn.BatchNorm2d(oc))
    def forward(self, x):
        return self.relu(self.b2(self.c2(self.relu(self.b1(self.c1(x))))) + self.skip(x))

class TinyResNet18(nn.Module):
    def __init__(self, num_classes=200, width_mult=0.48, dropout=0.2):
        super().__init__()
        c = [int(64*width_mult), int(128*width_mult), int(256*width_mult), int(512*width_mult)]
        self.stem   = nn.Sequential(nn.Conv2d(3,c[0],3,1,1,bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(BasicBlock(c[0],c[0]), BasicBlock(c[0],c[0]))
        self.layer2 = nn.Sequential(BasicBlock(c[0],c[1],2), BasicBlock(c[1],c[1]))
        self.layer3 = nn.Sequential(BasicBlock(c[1],c[2],2), BasicBlock(c[2],c[2]))
        self.layer4 = nn.Sequential(BasicBlock(c[2],c[3],2), BasicBlock(c[3],c[3]))
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Sequential(nn.Dropout(dropout), nn.Linear(c[3], num_classes))
    def forward(self, x):
        return self.fc(torch.flatten(self.gap(self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))),1))

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
    tr=transforms.Compose([
        transforms.RandomCrop(64,padding=8), transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2,magnitude=7),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25),
    ])
    vl=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
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

def train_one_epoch(model, loader, optimizer, criterion, cfg):
    model.train()
    tl = cor = tot = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        r = np.random.rand()
        if r < 0.25: imgs, la, lb, lam = cutmix(imgs, labels, cfg['cutmix_alpha'])
        elif r < 0.45: imgs, la, lb, lam = mixup(imgs, labels, cfg['mixup_alpha'])
        else: la, lb, lam = labels, labels, 1.0
        
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
        loss.backward()
        
        optimizer.first_step(zero_grad=False)
        
        with torch.amp.autocast('cuda'):
            loss2 = lam * criterion(model(imgs), la) + (1 - lam) * criterion(model(imgs), lb)
        
        optimizer.zero_grad() 
        loss2.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.second_step(zero_grad=True)
        
        tl += loss.item() * imgs.size(0)
        cor += (logits.argmax(1) == la).sum().item()
        tot += imgs.size(0)
        
    return tl / tot, cor / tot

@torch.no_grad()
def evaluate(model,loader):
    model.eval(); cor=tot=0
    for imgs,labels in loader:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
        cor+=(model(imgs).argmax(1)==labels).sum().item(); tot+=imgs.size(0)
    return cor/tot

def main():
    cfg=CFG
    model=TinyResNet18(200,0.48,cfg['dropout']).to(DEVICE)
    p=sum(x.numel() for x in model.parameters())
    print(f'Parameters: {p:,}'); assert p<=5_000_000
    criterion=LabelSmoothingCE(200,cfg['label_smooth'])
    optimizer=SAM(model.parameters(),optim.AdamW,rho=cfg['rho'],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    def lr_lambda(e):
        w=cfg['warmup_epoch']
        if e<w: return (e+1)/w
        return 0.5*(1+math.cos(math.pi*(e-w)/(cfg['num_epochs']-w)))
    scheduler=optim.lr_scheduler.LambdaLR(optimizer.base_optimizer,lr_lambda)
    train_loader,val_loader=get_loaders(cfg)
    best_acc=0.0
    for epoch in range(cfg['num_epochs']):
        tr_loss,tr_acc=train_one_epoch(model,train_loader,optimizer,criterion,cfg)
        val_acc=evaluate(model,val_loader)
        scheduler.step()
        lr_now=optimizer.base_optimizer.param_groups[0]['lr']
        if (epoch+1)%10==0 or epoch==0:
            print(f'[{epoch+1:3d}/{cfg["num_epochs"]}] loss={tr_loss:.4f} tr={tr_acc:.4f} val={val_acc:.4f} lr={lr_now:.5f}')
        if val_acc>best_acc:
            best_acc=val_acc
            torch.save(model.state_dict(),'best_sam_resnet.pth')
            print(f'  ? best: {best_acc:.4f}')
    print(f'\n done. Best: {best_acc:.4f}')
    model.load_state_dict(torch.load('best_sam_resnet.pth',map_location='cpu',weights_only=True))
    model.eval().cpu()
    dummy=torch.randn(1,3,64,64)
    traced=torch.jit.trace(model,dummy)
    torch.jit.save(traced,cfg['output_pt'])
    loaded=torch.jit.load(cfg['output_pt'],map_location='cpu').eval()
    with torch.no_grad(): out=loaded(dummy)
    assert out.shape==(1,200)
    print(f'? Export OK ? {sum(p.numel() for p in loaded.parameters()):,} params')

if __name__=='__main__':
    main()
