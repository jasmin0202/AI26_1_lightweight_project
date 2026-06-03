import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os, timm

DEVICE = torch.device('cuda:0')

class TinyImageNetVal(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir = os.path.join(root, 'images')
        self.transform = transform
        self.samples = []
        with open(os.path.join(root, 'labels.txt')) as f:
            for line in f:
                if line.strip():
                    fn, cl = line.strip().split('\t')
                    self.samples.append((fn, int(cl)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fn, lb = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fn)).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, lb

val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_ds = TinyImageNetVal('./student_data/public_val', transform=val_tf)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

model = timm.create_model('convnext_tiny', pretrained=False, num_classes=200)
model.stem[0] = nn.Conv2d(3, model.stem[0].out_channels, 3, 1, 1, bias=False)
model.load_state_dict(torch.load('best_teacher_server_convnext.pth', map_location='cpu', weights_only=True))
model.eval().to(DEVICE)

correct = total = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total += imgs.size(0)

print(f'Teacher val acc: {correct/total:.4f}')