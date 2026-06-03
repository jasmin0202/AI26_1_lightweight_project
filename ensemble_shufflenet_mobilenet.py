import os, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image

DEVICE = torch.device('cuda:1')


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


class EnsembleModel(nn.Module):
    def __init__(self, m1: nn.Module, m2: nn.Module):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.m1(x) + self.m2(x)) * 0.5


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


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); cor=tot=0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        cor += (model(imgs).argmax(1) == labels).sum().item()
        tot += imgs.size(0)
    return cor / tot


def main():
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_ds = TinyImageNetVal('./student_data/public_val', transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    m1 = ShuffleNetV2(200, dropout=0.2)
    m2 = get_mobilenet(200)

    m1.load_state_dict(torch.load('best_shufflenet.pth', map_location='cpu', weights_only=True))
    m2.load_state_dict(torch.load('best_weights_auto_mobilenet.pth', map_location='cpu', weights_only=True))

    m1.eval().to(DEVICE)
    m2.eval().to(DEVICE)

    acc1 = evaluate(m1, val_loader)
    acc2 = evaluate(m2, val_loader)
    print(f'ShuffleNet : {acc1:.4f}')
    print(f'MobileNet  : {acc2:.4f}')

    ens = EnsembleModel(m1, m2).to(DEVICE)
    acc_ens = evaluate(ens, val_loader)
    print(f'Ensemble   : {acc_ens:.4f}')

    p1 = sum(p.numel() for p in m1.parameters())
    p2 = sum(p.numel() for p in m2.parameters())
    print(f'params: {p1+p2:,} (ShuffleNet {p1:,} + MobileNet {p2:,})')
    assert p1+p2 <= 5_000_000, f'params exceed: {p1+p2:,}'

    m1.cpu(); m2.cpu()
    ens_cpu = EnsembleModel(m1, m2).eval()
    dummy = torch.randn(1, 3, 64, 64)
    traced = torch.jit.trace(ens_cpu, dummy)
    torch.jit.save(traced, 'my_model_shuffle_mobile.pt')

    loaded = torch.jit.load('my_model_shuffle_mobile.pt', map_location='cpu').eval()
    with torch.no_grad(): out = loaded(dummy)
    assert out.shape == (1, 200)
    print(f'Export OK shape {tuple(out.shape)}')
    print('save: my_model_shuffle_mobile.pt')


if __name__ == '__main__':
    main()