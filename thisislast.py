import os, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image

DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

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

def get_mobilenet(num_classes=200):
    m = mobilenet_v3_large(weights=None, num_classes=num_classes, reduced_tail=True)
    m.features[0][0] = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
    return m

class WeightedEnsembleModel(nn.Module):
    def __init__(self, m1: nn.Module, m2: nn.Module, w1: float, w2: float):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.w1 = w1
        self.w2 = w2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m1(x) * self.w1 + self.m2(x) * self.w2

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

@torch.no_grad()
def evaluate_single_model(model, loader):
    model.eval()
    cor = tot = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        cor += (model(imgs).argmax(1) == labels).sum().item()
        tot += imgs.size(0)
    return cor / tot

def main():
    print("==========================================", flush=True)
    print("?? Performance-Based Weighted Ensemble Pipe Active", flush=True)
    print("==========================================", flush=True)

    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_root = './student_data'
    val_ds = TinyImageNetDataset(data_root, is_val=True, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    m1 = ShuffleNetV2(200, dropout=0.2).to(DEVICE)
    m2 = get_mobilenet(200).to(DEVICE)

    m1.load_state_dict(torch.load('best_shufflenet_kd_pure.pth', map_location=DEVICE, weights_only=True))
    m2.load_state_dict(torch.load('best_mobilenet_kd_pure.pth', map_location=DEVICE, weights_only=True))
    
    m1.eval(); m2.eval()

    print("\n>> Evaluating individual models to calculate optimal weights...")
    acc1 = evaluate_single_model(m1, val_loader)
    acc2 = evaluate_single_model(m2, val_loader)
    
    print(f" -> ShuffleNetV2 Base Accuracy: {acc1*100:.2f}%", flush=True)
    print(f" -> MobileNetV3  Base Accuracy: {acc2*100:.2f}%", flush=True)

    # Performance-based weight calculation
    sum_acc = acc1 + acc2
    w1 = acc1 / sum_acc
    w2 = acc2 / sum_acc
    print(f"\n>> Calculated Performance Weights -> Shuffle: {w1:.4f} | Mobile: {w2:.4f}")

    # Evaluate the performance-weighted ensemble
    ens_model = WeightedEnsembleModel(m1, m2, w1, w2)
    cor = tot = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            cor += (ens_model(imgs).argmax(1) == labels).sum().item()
            tot += imgs.size(0)
    
    final_acc = (cor / tot) * 100
    print("\n==========================================")
    print(f"?? Performance-Weighted Ensemble Val Acc: {final_acc:.2f}%")
    print("==========================================")

    # Export to TorchScript (.pt)
    m1.cpu(); m2.cpu()
    final_ens_cpu = WeightedEnsembleModel(m1, m2, w1, w2).eval()
    
    dummy = torch.randn(1, 3, 64, 64)
    traced = torch.jit.trace(final_ens_cpu, dummy)

    output_filename = 'my_model_perf_weighted_ensemble.pt'
    torch.jit.save(traced, output_filename)
    print(f'>> [SUCCESS] Exported Performance-Weighted file to: {output_filename}', flush=True)

if __name__ == '__main__':
    main()