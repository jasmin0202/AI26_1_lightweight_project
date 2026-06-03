import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

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
                    
    def __len__(self): 
        return len(self.samples)
        
    def __getitem__(self, idx):
        fn, lb = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fn)).convert('RGB')
        if self.transform: 
            img = self.transform(img)
        return img, lb

@torch.no_grad()
def evaluate_jit(model, loader):
    model.eval()
    cor = tot = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        cor += (outputs.argmax(1) == labels).sum().item()
        tot += imgs.size(0)
    return cor / tot

def main():
    print("==================================================")
    print("?? Starting validation for the finalized ensemble model...")
    print("==================================================")

    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_ds = TinyImageNetVal('./student_data/public_val', transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=4, pin_memory=True)

    target_file = 'my_model_shuffle_mobile.pt'
    if not os.path.exists(target_file):
        print(f"?? Error: {target_file} not found in the current directory!")
        return

    print(f">> 1. Loading {target_file}...")
    try:
        model = torch.jit.load(target_file, map_location=DEVICE).eval()
        print(">> [SUCCESS] TorchScript ensemble model loaded successfully.")
    except Exception as e:
        print(f"?? Critical Error during model loading: {e}")
        return

    print(">> 2. Running time trial on Public Validation set...")
    final_acc = evaluate_jit(model, val_loader)

    print("\n==================================================")
    print(f"?? Final Ensemble Model Validation Result")
    print(f"?? Accuracy for {target_file}: {final_acc * 100:.2f}%")
    print("==================================================")

if __name__ == '__main__':
    main()