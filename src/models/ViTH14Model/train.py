import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import albumentations as A
import albumentations.pytorch as APT
from torch.utils.data import DataLoader
from torchvision.models import vit_h_14, ViT_H_14_Weights
from sklearn.model_selection import train_test_split

from models.ViTH14Model.Dataset import ViTH14Dataset

from scripts.train import train as train_model
from utils.set_seed import set_seed


def train(batch_size = 16, learning_rate = 1e-05, num_epochs = 16, seed = 42, lr_min = 1e-06, flozen = False):

    set_seed(seed)

    transform = {
        "train": A.Compose([
            A.Resize(518, 518),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.GaussianBlur(blur_limit=(3, 3), p=0.05),
            A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.3),
            A.Normalize(),
            APT.ToTensorV2()
        ]),
        "val": A.Compose([
            A.Resize(518, 518),
            A.Normalize(),
            APT.ToTensorV2()
        ]),
    }

    train_file_path = os.path.join(os.path.dirname(__file__), '../../data/input/train.csv')
    train_img_fd_path = os.path.join(os.path.dirname(__file__), '../../data/input/images/train')
    model_save_path = os.path.join(os.path.dirname(__file__), '../../data/models/ViTH14Model')

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    train_master = pd.read_csv(train_file_path)
    image_name_list = train_master['image_name'].values
    label_list = train_master['label'].values

    X_train, X_val, y_train, y_val = train_test_split(image_name_list, label_list, test_size=0.5, stratify=label_list, random_state=seed)

    train_dataset = ViTH14Dataset(X_train, y_train, root_dir=train_img_fd_path, transform=transform, phase='train')
    val_dataset = ViTH14Dataset(X_val, y_val, root_dir=train_img_fd_path, transform=transform, phase='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }

    model = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
    model.heads[0] = nn.Linear(in_features=1280, out_features=2, bias=True)
    
    if flozen:
        layer_count = 0
        for param in model.parameters():
            param.requires_grad = False
            layer_count += 1
            if layer_count >= 150:
                break

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=lr_min)

    best_model_path, loss_history, auc_history, _, _ = train_model(
        model, num_epochs, criterion, optimizer, dataloaders, scheduler,
        model_save_path = model_save_path
    )

    return best_model_path, loss_history, auc_history