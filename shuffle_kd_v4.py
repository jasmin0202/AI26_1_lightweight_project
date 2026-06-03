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
print(f'GPU: {torch.cuda.get_device_name(2)}')

CFG = dict(
    data_root       = './student_data',
    batch_size      = 256,
    num_epochs      = 25,            # Finishes within 1 hour 40 mins
    lr              = 3e-4,          # Increased LR to force escape from local minima
    weight_decay    = 0.01,
    label_smooth    = 0.1,
    dropout         = 0.3,
    num_workers     = 4,
    kd_temperature  = 3.0,          # Slightly lower temperature for sharper guidance
    kd_alpha        = 1.0,          # 100% pure KD, blocking hard label memorization
    student_ckpt    = 'best_shufflenet_ext.pth',
    teacher_ckpt    = 'best_teacher_resnet50_v2.pth',
    output_pth      = 'best_shufflenet_kd_pure.pth',
    output_pt       = 'my_model_shufflenet_kd_pure.pt',
)


def get_resnet50(num_classes=200, dropout=0.3):
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, num_classes))
    return model


def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)


class ShuffleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.stride = stride
        branch_ch = out_ch // 2
        if stride == 1:
            self.branch2 = nn.Sequential(
                nn.Conv2d(branch_ch, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
                nn.Conv2d(branch_ch, branch_ch, 3, 1, 1, groups=branch_ch, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.Conv2d(branch_ch, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
                nn.Conv2d(branch_ch, branch_ch, 3, stride, 1, groups=branch_ch, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.Conv2d(branch_ch, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            return channel_shuffle(torch.cat([x1, self.branch2(x2)], dim=1), 2)
        return channel_shuffle(torch.cat([self.branch1(x), self.branch2(x)], dim=1), 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=200, dropout=0.2):
        super().__init__()
        ch = [32, 176, 352, 704, 1024]

        self.stem = nn.Sequential(
            nn.Conv2d(3, ch[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch[0]), nn.ReLU(inplace=True),
        )
        self.stage2 = self._make_stage(ch[0], ch[1], n=4)
        self.stage3 = self._make_stage(ch[1], ch[2], n=8)
        self.stage4 = self._make_stage(ch[2], ch[3], n=4)
        self.conv5 = nn.Sequential(
            nn.Conv2d(ch[3], ch[4], 1, bias=False),
            nn.BatchNorm2d(ch[4]), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch[4], num_classes))

    def _make_stage(self, in_ch, out_ch, n):
        layers = [ShuffleBlock(in_ch, out_ch, stride=2)]
        for _ in range(n-1):
            layers.append(ShuffleBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return self.fc(torch.flatten(self.gap(x), 1))


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

    student = ShuffleNetV2(200, dropout=cfg['dropout']).to(DEVICE)
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