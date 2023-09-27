import os
import timm
import pandas as pd
import torch.nn as nn
from ranger21 import Ranger21
import torch.optim.lr_scheduler as lr_scheduler
import albumentations as A
import albumentations.pytorch as APT
from torch.utils.data import DataLoader
from torchvision.models.swin_transformer import swin_v2_b, Swin_V2_B_Weights
import torch.optim as optim
from sklearn.model_selection import train_test_split

from models.SwinV2Model.Dataset import SwinV2Dataset

from scripts.train import train as train_model
from utils.set_seed import set_seed


def train(batch_size = 16, learning_rate = 1e-05, num_epochs = 16, seed = 42, lr_min = 1e-06, model_name = "swinv2_large_window12to24_192to384", image_size = 384, pretrained = True, use_flozen = True):

    set_seed(seed)

    transform = {
        "train": A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.9, 1.0), ratio=(0.8, 1.2)),
            A.HorizontalFlip(p=0.35),
            A.VerticalFlip(p=0.35),
            A.GaussianBlur(blur_limit=(9, 11), p=0.1),
            A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.3),
            A.Normalize(),
            APT.ToTensorV2()
        ]),
        "val": A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(),
            APT.ToTensorV2()
        ]),
    }

    train_file_path = os.path.join(os.path.dirname(__file__), '../../data/input/train.csv')
    train_img_fd_path = os.path.join(os.path.dirname(__file__), '../../data/input/images/train')
    model_save_path = os.path.join(os.path.dirname(__file__), '../../data/models/SwinV2Model')

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    train_master = pd.read_csv(train_file_path)
    image_name_list = train_master['image_name'].values
    label_list = train_master['label'].values

    X_train, X_val, y_train, y_val = train_test_split(image_name_list, label_list, test_size=0.5, stratify=label_list, random_state=seed)

    train_dataset = SwinV2Dataset(X_train, y_train, root_dir=train_img_fd_path, transform=transform, phase='train')
    val_dataset = SwinV2Dataset(X_val, y_val, root_dir=train_img_fd_path, transform=transform, phase='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=2)
    if pretrained and use_flozen:
        layer_count = 0
        for param in model.parameters():
            param.requires_grad = False
            layer_count += 1
            if layer_count == 200:
                break

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=lr_min)

    best_model_path, loss_history, auc_history, _, _ = train_model(
        model, num_epochs, criterion, optimizer, dataloaders, scheduler,
        model_save_path = model_save_path
    )

    return best_model_path, loss_history, auc_history