import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import albumentations as A
import albumentations.pytorch as APT
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.PackageModel.Net import PackageNet
from models.PackageModel.Dataset import PackageDataset

from scripts.train import train as train_model
from utils.set_seed import set_seed


def train(batch_size = 32, learning_rate = 1e-03, num_epochs = 16, seed = 42):

    set_seed(seed)

    transform = {
        "train": A.Compose([
            A.Resize(128, 128),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(blur_limit=(9, 13), p=0.25),
            A.JpegCompression(quality_lower=75, quality_upper=100, p=0.25),
            A.Normalize(),
            APT.ToTensorV2()
        ]),
        "val": A.Compose([
            A.Resize(128, 128),
            A.Normalize(),
            APT.ToTensorV2()
        ]),
    }

    train_file_path = os.path.join(os.path.dirname(__file__), '../../data/input/train.csv')
    train_img_fd_path = os.path.join(os.path.dirname(__file__), '../../data/input/images/train')
    model_save_path = os.path.join(os.path.dirname(__file__), '../../data/models/PackageModel')

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    train_master = pd.read_csv(train_file_path)
    image_name_list = train_master['image_name'].values
    label_list = train_master['label'].values

    X_train, X_val, y_train, y_val = train_test_split(image_name_list, label_list, test_size=0.5, stratify=label_list, random_state=seed)

    train_dataset = PackageDataset(X_train, y_train, root_dir=train_img_fd_path, transform=transform, phase='train')
    val_dataset = PackageDataset(X_val, y_val, root_dir=train_img_fd_path, transform=transform, phase='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }

    model = PackageNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_model_path, loss_history, auc_history = train_model(
        model, num_epochs, criterion, optimizer, dataloaders,
        model_save_path = model_save_path
    )

    return best_model_path, loss_history, auc_history