import os, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image

DEVICE = torch.device('cuda:1')

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_ch,out_ch,3,stride,1,bias=False); self.bn1=nn.BatchNorm2d(out_ch)
        self.conv2=nn.Conv2d(out_ch,out_ch,3,1,1,bias=False); self.bn2=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU(inplace=True)
        self.shortcut=nn.Sequential()
        if stride!=1 or in_ch!=out_ch:
            self.shortcut=nn.Sequential(nn.Conv2d(in_ch,out_ch,1,stride,bias=False),nn.BatchNorm2d(out_ch))
    def forward(self,x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))+self.shortcut(x))

class TinyResNet18(nn.Module):
    def __init__(self, num_classes=200, width_mult=0.48, dropout=0.2):
        super().__init__()
        c=[int(64*width_mult),int(128*width_mult),int(256*width_mult),int(512*width_mult)]
        self.stem=nn.Sequential(nn.Conv2d(3,c[0],3,1,1,bias=False),nn.BatchNorm2d(c[0]),nn.ReLU(inplace=True))
        self.layer1=nn.Sequential(BasicBlock(c[0],c[0]),BasicBlock(c[0],c[0]))
        self.layer2=nn.Sequential(BasicBlock(c[0],c[1],2),BasicBlock(c[1],c[1]))
        self.layer3=nn.Sequential(BasicBlock(c[1],c[2],2),BasicBlock(c[2],c[2]))
        self.layer4=nn.Sequential(BasicBlock(c[2],c[3],2),BasicBlock(c[3],c[3]))
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Dropout(dropout),nn.Linear(c[3],num_classes))
    def forward(self,x):
        return self.fc(torch.flatten(self.gap(self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))),1))

def get_mobilenet(num_classes=200):
    m=mobilenet_v3_large(weights=None,num_classes=num_classes,reduced_tail=True)
    m.features[0][0]=nn.Conv2d(3,16,3,1,1,bias=False)
    return m

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

class EnsembleTTA(nn.Module):
    def __init__(self, m1: nn.Module, m2: nn.Module):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits  = self.m1(x) + self.m2(x)
        logits += self.m1(torch.flip(x, [3])) + self.m2(torch.flip(x, [3]))
        logits += self.m1(torch.flip(x, [2])) + self.m2(torch.flip(x, [2]))
        return logits / 6.0

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); cor=tot=0
    for imgs,labels in loader:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
        cor+=(model(imgs).argmax(1)==labels).sum().item(); tot+=imgs.size(0)
    return cor/tot

def main():
    val_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    val_ds=TinyImageNetVal('./student_data/public_val',transform=val_tf)
    val_loader=DataLoader(val_ds,batch_size=128,shuffle=False,num_workers=4,pin_memory=True)

    m1=TinyResNet18(200,0.48,0.2)
    m2=get_mobilenet(200)

    # 원본 vs SAM ft 비교
    for tag, r_path, m_path in [
    ('원본',              'best_weights_auto_resnet.pth',  'best_weights_auto_mobilenet.pth'),
    ('SAM ft ResNet만',  'sam_ft_resnet.pth',              'best_weights_auto_mobilenet.pth'),
]:
        if not os.path.exists(r_path) or not os.path.exists(m_path):
            print(f'{tag}: 파일 없음, 스킵')
            continue
        m1.load_state_dict(torch.load(r_path, map_location='cpu', weights_only=True))
        m2.load_state_dict(torch.load(m_path, map_location='cpu', weights_only=True))
        m1.eval().to(DEVICE); m2.eval().to(DEVICE)

        acc_no_tta = evaluate(EnsembleTTA(m1,m2), val_loader)

        ens = EnsembleTTA(m1, m2).to(DEVICE)
        acc_tta = evaluate(ens, val_loader)
        print(f'[{tag}] TTA val acc: {acc_tta:.4f}  (no-TTA: {acc_no_tta:.4f})')

    # export
    print('\nbest part export...')
    best_tag = None
    best_acc = 0.0
    best_paths = None

    for tag, r_path, m_path in [
    ('원본',              'best_weights_auto_resnet.pth',  'best_weights_auto_mobilenet.pth'),
    ('SAM ft ResNet만',  'sam_ft_resnet.pth',              'best_weights_auto_mobilenet.pth'),
]:
        if not os.path.exists(r_path) or not os.path.exists(m_path):
            continue
        m1.load_state_dict(torch.load(r_path, map_location='cpu', weights_only=True))
        m2.load_state_dict(torch.load(m_path, map_location='cpu', weights_only=True))
        m1.eval().to(DEVICE); m2.eval().to(DEVICE)
        ens = EnsembleTTA(m1, m2).to(DEVICE)
        acc = evaluate(ens, val_loader)
        if acc > best_acc:
            best_acc = acc
            best_tag = tag
            best_paths = (r_path, m_path)

    print(f'best: {best_tag} ({best_acc:.4f})')
    m1.load_state_dict(torch.load(best_paths[0], map_location='cpu', weights_only=True))
    m2.load_state_dict(torch.load(best_paths[1], map_location='cpu', weights_only=True))
    m1.eval().cpu(); m2.eval().cpu()

    ens = EnsembleTTA(m1, m2).eval()

    p_total = sum(p.numel() for p in m1.parameters()) + sum(p.numel() for p in m2.parameters())
    print(f'params: {p_total:,}')
    assert p_total <= 5_000_000, f'params exceed: {p_total:,}'

    dummy = torch.randn(1, 3, 64, 64)
    traced = torch.jit.trace(ens, dummy)
    torch.jit.save(traced, 'my_model_tta_ensemble.pt')

    loaded = torch.jit.load('my_model_tta_ensemble.pt', map_location='cpu').eval()
    with torch.no_grad(): out = loaded(dummy)
    assert out.shape == (1, 200)
    print(f'Export OK shape {tuple(out.shape)}')
    print('저장: my_model_tta_ensemble.pt')

if __name__ == '__main__':
    main()
