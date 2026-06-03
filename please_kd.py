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
import timm

SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

CFG = dict(
    data_root    = './student_data',
    batch_size   = 256,
    num_epochs   = 15,
    lr           = 2e-5,
    weight_decay = 0.05,
    label_smooth = 0.1,
    dropout      = 0.2,
    num_workers  = 4,
    cutmix_alpha = 1.0,
    mixup_alpha  = 0.2,
    output_pt    = 'my_model_resnet18.pt'
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

def train_kd_epoch(teacher, student, loader, optimizer, temperature=4.0, alpha=0.4):
    teacher.eval()
    student.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.no_grad():
            teacher_logits = teacher(imgs)
            
        with torch.amp.autocast('cuda'):
            student_logits = student(imgs)
            loss_ce = F.cross_entropy(student_logits, labels, label_smoothing=CFG['label_smooth'])
            loss_kd = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)
            loss = alpha * loss_ce + (1 - alpha) * loss_kd
            
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return correct / total

if __name__ == '__main__':
    train_loader, val_loader = get_loaders(CFG)
    
    print(">> Loading Teacher model via exact configurations...")
    teacher = timm.create_model('convnext_tiny', pretrained=False, num_classes=200, drop_path_rate=0.3).to(DEVICE)
    teacher.stem[0] = nn.Conv2d(3, teacher.stem[0].out_channels, kernel_size=3, stride=1, padding=1, bias=False).to(DEVICE)
    
    teacher_state = torch.load('best_teacher_server_convnext.pth', map_location=DEVICE, weights_only=True)
    teacher.load_state_dict(teacher_state)
    print(f">> Teacher Baseline Val Acc: {evaluate(teacher, val_loader):.4f}")

    print(">> Loading pre-trained Student checkpoints...")
    m1 = TinyResNet18(num_classes=200, width_mult=0.48, dropout=CFG['dropout']).to(DEVICE)
    m2 = get_mobilenet_v3(num_classes=200).to(DEVICE)
    
    m1.load_state_dict(torch.load('best_weights_auto_resnet.pth', map_location=DEVICE, weights_only=True))
    m2.load_state_dict(torch.load('best_weights_auto_mobilenet.pth', map_location=DEVICE, weights_only=True))
    
    print(f">> Initial ResNet18 Val Acc: {evaluate(m1, val_loader):.4f}")
    print(f">> Initial MobileNetV3 Val Acc: {evaluate(m2, val_loader):.4f}")

    opt_res = optim.AdamW(m1.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    opt_mobile = optim.AdamW(m2.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])

    print(">> Training ResNet18 via KD (15 epochs)...")
    for epoch in range(CFG['num_epochs']):
        loss = train_kd_epoch(teacher, m1, train_loader, opt_res)
        val_acc = evaluate(m1, val_loader)
        print(f"  [Epoch {epoch+1:2d}] Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

    print(">> Training MobileNetV3 via KD (15 epochs)...")
    for epoch in range(CFG['num_epochs']):
        loss = train_kd_epoch(teacher, m2, train_loader, opt_mobile)
        val_acc = evaluate(m2, val_loader)
        print(f"  [Epoch {epoch+1:2d}] Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

    print(">> Exporting Ensemble Wrapper to TorchScript...")
    ensemble_model = EnsembleWrapper(m1.cpu(), m2.cpu()).eval()
    
    dummy_input = torch.randn(1, 3, 64, 64)
    traced_model = torch.jit.trace(ensemble_model, dummy_input)
    torch.jit.save(traced_model, CFG['output_pt'])
    
    print(f">> [SUCCESS] Saved final submission file: '{CFG['output_pt']}'")