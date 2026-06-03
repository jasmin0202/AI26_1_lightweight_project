import os, random, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda:1') # Change to your available second GPU index


def get_mobilenet(num_classes=200):
    m = mobilenet_v3_large(weights=None, num_classes=num_classes, reduced_tail=True)
    m.features[0][0] = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
    return m


class TinyImageNetDataset(Dataset):
    def __init__(self, root, is_val=False, transform=None):
        self.transform = transform
        self.samples = []
        if is_val:
            self.img_dir = os.path.join(root, 'public_val', 'images')
            lbl_path = os.path.join(root, 'public_val', 'labels.txt')
            with open(lbl_path) as f:
                for line in f:
                    if line.strip():
                        fn, cl = line.strip().split('\t')
                        self.samples.append((os.path.join(self.img_dir, fn), int(cl)))
        else:
            wnids_path = os.path.join(root, 'wnids.txt')
            with open(wnids_path) as f:
                wnids = [line.strip() for line in f if line.strip()]
            class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
            self.train_dir = os.path.join(root, 'train')
            for wnid in wnids:
                images_dir = os.path.join(self.train_dir, wnid, 'images')
                if os.path.exists(images_dir):
                    cl_idx = class_to_idx[wnid]
                    for fn in os.listdir(images_dir):
                        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(images_dir, fn), cl_idx))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, lb = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, lb


def main():
    # 64x64 Safe Mild Augmentation ONLY
    train_tf = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_root = './student_data'
    train_ds = TinyImageNetDataset(data_root, is_val=False, transform=train_tf)
    val_ds = TinyImageNetDataset(data_root, is_val=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    model = get_mobilenet(200).to(DEVICE)
    model.load_state_dict(torch.load('best_weights_auto_mobilenet.pth', map_location=DEVICE, weights_only=True))

    epochs = 20
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(">> Starting Safe Mild Refinement for MobileNetV3...", flush=True)
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        cor = tot = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                cor += (model(imgs).argmax(1) == labels).sum().item()
                tot += imgs.size(0)
        val_acc = (cor / tot) * 100
        print(f"[MobileNetV3] Epoch {epoch+1:02d}/{epochs} | Val Acc: {val_acc:.2f}%", flush=True)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_advanced_mobilenet.pth')
            print(f"  => [SAVE] New Best MobileNet Saved: {best_acc:.2f}%", flush=True)


if __name__ == '__main__':
    main()