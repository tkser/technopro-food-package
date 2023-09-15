import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import albumentations as A
import albumentations.pytorch as APT
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6
from torchvision.models import EfficientNet_B2_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights
from sklearn.model_selection import train_test_split

from models.EfficientNetModel.Dataset import EfficientNetDataset

from scripts.train import train as train_model
from utils.set_seed import set_seed


def train(batch_size = 16, learning_rate = 1e-05, num_epochs = 16, seed = 42, lr_min = 1e-06, flozen = False, model_type = 'b3'):

    set_seed(seed)

    image_size = 0
    if model_type == 'b2':
        image_size = 288
    elif model_type == 'b3':
        image_size = 320
    elif model_type == 'b4':
        image_size = 384
    elif model_type == 'b5':
        image_size = 456
    elif model_type == 'b6':
        image_size = 528
    
    if image_size == 0:
        raise ValueError('Invalid model_type')

    transform = {
        "train": A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.GaussianBlur(blur_limit=(3, 3), p=0.05),
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
    model_save_path = os.path.join(os.path.dirname(__file__), '../../data/models/EfficientNetModel', model_type)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    train_master = pd.read_csv(train_file_path)
    image_name_list = train_master['image_name'].values
    label_list = train_master['label'].values

    X_train, X_val, y_train, y_val = train_test_split(image_name_list, label_list, test_size=0.5, stratify=label_list, random_state=seed)

    train_dataset = EfficientNetDataset(X_train, y_train, root_dir=train_img_fd_path, transform=transform, phase='train')
    val_dataset = EfficientNetDataset(X_val, y_val, root_dir=train_img_fd_path, transform=transform, phase='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }

    model = None

    if model_type == 'b2':
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    elif model_type == 'b3':
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    elif model_type == 'b4':
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    elif model_type == 'b5':
        model = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
    elif model_type == 'b6':
        model = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
    
    if model is None:
        raise ValueError('Invalid model_type')

    if flozen:
        for param in model.parameters():
            param.requires_grad = False
    model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=2)
    if flozen:
        for param in model.classifier.parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=lr_min)

    best_model_path, loss_history, auc_history, _, _ = train_model(
        model, num_epochs, criterion, optimizer, dataloaders, scheduler,
        model_save_path = model_save_path
    )

    return best_model_path, loss_history, auc_history