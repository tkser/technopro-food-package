import os
from sklearn.metrics import roc_auc_score
import timm
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import albumentations as A
import albumentations.pytorch as APT
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

from models.SwinV2Model.Dataset import SwinV2Dataset

from utils.logger import logger
from scripts.train import train as train_model
from utils.set_seed import set_seed


def train_cv(batch_size = 16, learning_rate = 1e-05, num_epochs = 16, seed = 42, lr_min = 1e-06, model_name = "swinv2_large_window12to24_192to384", n_splits = 5, pretrained = True, use_flozen = True, start_fold = 0):

    set_seed(seed)

    transform = {
        "train": A.Compose([
            A.Resize(384, 384),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(blur_limit=(9, 11), p=0.3),
            A.Normalize(),
            APT.ToTensorV2()
        ]),
        "val": A.Compose([
            A.Resize(384, 384),
            A.Normalize(),
            APT.ToTensorV2()
        ]),
    }

    train_file_path = os.path.join(os.path.dirname(__file__), '../../data/input/train.csv')
    train_img_fd_path = os.path.join(os.path.dirname(__file__), '../../data/input/images/train')
    model_save_path = os.path.join(os.path.dirname(__file__), '../../data/models/SwinV2Model_cv')

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    train_master = pd.read_csv(train_file_path)
    image_name_list = train_master['image_name'].values
    label_list = train_master['label'].values

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_pred = []
    cv_true = []
    best_model_paths = []
    loss_histories = []
    auc_histories = []

    for fold, (train_index, val_index) in enumerate(kf.split(image_name_list, label_list)):

        if fold < start_fold:
            continue
        
        logger.debug(f"Fold: {fold + 1}/{n_splits}")
        logger.debug(f"--" * 20)
        
        X_train, X_val = image_name_list[train_index], image_name_list[val_index]
        y_train, y_val = label_list[train_index], label_list[val_index]

        train_dataset = SwinV2Dataset(X_train, y_train, root_dir=train_img_fd_path, transform=transform, phase='train')
        val_dataset = SwinV2Dataset(X_val, y_val, root_dir=train_img_fd_path, transform=transform, phase='val')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        dataloaders = {
            "train": train_loader,
            "val": val_loader
        }

        model = timm.create_model(model_name, pretrained=True, num_classes=2)
        if pretrained and use_flozen:
            for param in model.parameters():
                param.requires_grad = False
            
            model.head.fc = nn.Linear(model.head.fc.in_features, 2, bias=True)
            for param in model.head.fc.parameters():
                param.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=lr_min)

        best_model_path, loss_history, auc_history, best_auc_pred_list, best_auc_true_list = train_model(
            model, num_epochs, criterion, optimizer, dataloaders, scheduler,
            model_save_path = model_save_path
        )

        cv_pred.extend(best_auc_pred_list)
        cv_true.extend(best_auc_true_list)
        best_model_paths.append(best_model_path)
        loss_histories.append(loss_history)
        auc_histories.append(auc_history)

        logger.debug(f"Best model path: {best_model_path}")
        logger.debug(f"Best AUC: {auc_history['val'][-1]}")
        logger.debug(f"Loss history: {loss_history}")
        logger.debug(f"AUC history: {auc_history}")
        logger.debug(f"--" * 20)
    
    loss_history_avg = {'train': [], 'val': []}
    auc_history_avg = {'train': [], 'val': []}

    for loss_history in loss_histories:
        for key in loss_history_avg.keys():
            loss_history_avg[key].append(loss_history[key])
    for auc_history in auc_histories:
        for key in auc_history_avg.keys():
            auc_history_avg[key].append(auc_history[key])
    
    for key in loss_history_avg.keys():
        loss_history_avg[key] = np.mean(loss_history_avg[key], axis=0)
    for key in auc_history_avg.keys():
        auc_history_avg[key] = np.mean(auc_history_avg[key], axis=0)

    logger.debug(f"CV AUC: {roc_auc_score(cv_true, cv_pred)}")
    logger.debug(f"Best model paths: {best_model_paths}")

    return best_model_paths, loss_history_avg, auc_history_avg