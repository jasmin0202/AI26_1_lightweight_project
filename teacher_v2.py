import os
import sys
import time
import math
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


import timm
from timm.data.mixup import Mixup

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(777)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("server_training_diagnostic.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

CFG = dict(
    data_root    = './student_data',
    batch_size   = 128,          
    num_epochs   = 150,          
    warmup_epoch = 5,            
    lr           = 5e-4,         
    weight_decay = 0.05,
    label_smooth = 0.1,
    dropout      = 0.3,          
    num_workers  = 4,            
    cutmix_alpha = 0.5,          
    mixup_alpha  = 0.5,          
    save_name    = 'best_teacher_server_convnext.pth', 
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_perfect_teacher(num_classes=200):
    logger.info("Building server ConvNeXt-Tiny model skeleton...")
    model = timm.create_model(
        'convnext_tiny', 
        pretrained=False,  
        num_classes=num_classes,
        drop_path_rate=CFG['dropout']
    )
    
    model.stem[0] = nn.Conv2d(
        in_channels=3, 
        out_channels=model.stem[0].out_channels, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=False
    )
    return model

def get_data_loaders(cfg):
    logger.info("Setting up dataset and DataLoader...")
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        train_dir = os.path.join(cfg['data_root'], 'train')
        val_dir = os.path.join(cfg['data_root'], 'public_val')
        
        from torchvision.datasets import ImageFolder
        class TinyImageNetTrain(ImageFolder):
            def __init__(self, root, transform=None):
                super().__init__(root, transform=transform)
            def _find_classes(self, dir):
                classes = sorted(entry.name for entry in os.scandir(dir) if entry.is_dir())
                class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
                return classes, class_to_idx
                
        train_dataset = TinyImageNetTrain(root=train_dir, transform=transform_train)
        val_dataset = ImageFolder(root=val_dir, transform=transform_val)
        
        logger.info(f"Dataset load complete. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    except Exception as e:
        logger.error(f"Dataset path error: {e}")
        raise e

    train_loader = DataLoader(
        train_dataset, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=True
    )
    
    return train_loader, val_loader

def get_criterion_and_mixup(cfg):
    mixup_fn = Mixup(
        mixup_alpha=cfg['mixup_alpha'], cutmix_alpha=cfg['cutmix_alpha'],
        label_smoothing=cfg['label_smooth'], num_classes=200
    )
    criterion = nn.CrossEntropyLoss()
    return criterion, mixup_fn

def main():
    logger.info(f"Pipeline active. Device: {DEVICE}")
    
    model = get_perfect_teacher(num_classes=200).to(DEVICE)
    train_loader, val_loader = get_data_loaders(CFG)
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['num_epochs'] - CFG['warmup_epoch'], eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    criterion, mixup_fn = get_criterion_and_mixup(CFG)
    
    try:
        logger.info("Testing initial batch procurement...")
        iterator = iter(train_loader)
        _ = next(iterator)
        logger.info("Data channel initialized successfully.")
    except Exception as e:
        logger.critical(f"Initial loading error: {e}")
        sys.exit(1)

    for epoch in range(1, CFG['num_epochs'] + 1):
        logger.info(f"Epoch [{epoch}/{CFG['num_epochs']}] starting...")
        model.train()
        
        if epoch <= CFG['warmup_epoch']:
            current_lr = CFG['lr'] * (epoch / CFG['warmup_epoch'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            if epoch > CFG['warmup_epoch']:
                imgs, labels_mixed = mixup_fn(imgs, labels)
            else:
                labels_mixed = labels
                
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                if epoch > CFG['warmup_epoch']:
                    loss = criterion(outputs, labels_mixed)
                else:
                    loss = criterion(outputs, labels)
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"  [Batch {batch_idx + 1}/{len(train_loader)}] loss: {loss.item():.4f}")
                
        epoch_time = time.time() - start_time
        logger.info(f" Epoch {epoch} done. Time: {epoch_time/60:.2f}m, Average Loss: {total_loss/len(train_loader):.4f}")
        
        if epoch > CFG['warmup_epoch']:
            scheduler.step()

        torch.save(model.state_dict(), CFG['save_name'])

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    logger.info("Server teacher B model process initialized.")
    main()