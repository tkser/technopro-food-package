import os
import gc
import datetime
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from utils.logger import logger
from utils.rocstar import roc_star_loss, epoch_update_gamma

from typing import Dict, Tuple, Callable, Optional


def train(
        net: nn.Module,
        epochs: int,
        loss_fn: Callable,
        optimizer: optim.Optimizer,
        dataloaders: Dict[str, DataLoader],
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        model_save_path: Optional[str] = None
    ) -> Tuple[Optional[str], Dict[str, list], Dict[str, list]] :

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net.to(device)
    logger.debug(f"Using device: {device}({torch.cuda.get_device_name()})")

    tz = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(tz)

    logger.debug(f"Starting training on {device} at {now.strftime('%Y-%m-%d %H:%M:%S')}")

    logger.debug(f"Model parameters:")
    for name, param in net.named_parameters():
        logger.debug(f"{name}: {param.shape}")
    
    logger.debug(f"Optimizer: {optimizer}")
    logger.debug(f"Scheduler: {scheduler}")
    logger.debug(f"Loss function: {roc_star_loss}")

    best_auc = 0.0
    best_auc_model_path = None

    loss_history = {'train': [], 'val': []}
    auc_history = {'train': [], 'val': []}

    last_epoch_y_pred = None
    last_epoch_y_true = None
    epoch_gamma = 0.0

    for epoch in range(epochs):

        logger.debug(f'Epoch: {epoch+1} / {epochs}')
        logger.debug('--------------------------')

        for phase in ['train', 'val']:

            if phase == 'train':
                net.train()
            else:
                net.eval()
        
            epoch_loss = 0.0
            pred_list = []
            true_list = []

            for images, labels in tqdm(dataloaders[phase], desc=f"Epoch {epoch+1}/{epochs}"):

                images = images.float().to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):

                    outputs = net(images)
                    if epoch == 0:
                        loss = loss_fn(outputs, labels)
                    else:
                        loss = roc_star_loss(labels, outputs, epoch_gamma, last_epoch_y_pred, last_epoch_y_true)
                    preds = torch.softmax(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                    
                    epoch_loss += np.mean(loss.detach().cpu().numpy()) * images.size(0)

                    preds = preds.detach().to('cpu').numpy()
                    pred_list.append(preds)

                    labels = labels.detach().to('cpu').numpy()
                    true_list.extend(labels)
                
                del images, labels, outputs, preds
                torch.cuda.empty_cache()

            epoch_loss = epoch_loss / len(dataloaders[phase].dataset)

            pred_list = np.concatenate(pred_list, axis=0)
            pred_list = pred_list[:, 1]

            if phase == 'train':
                last_epoch_y_pred = torch.tensor(pred_list,dtype=torch.float).to(device)
                last_epoch_y_true = torch.tensor(true_list,dtype=torch.float).to(device)
                epoch_gamma = epoch_update_gamma(last_epoch_y_true, last_epoch_y_pred, epoch, 2)
            
            epoch_auc = roc_auc_score(true_list, pred_list)

            loss_history[phase].append(epoch_loss)
            auc_history[phase].append(epoch_auc)

            logger.debug(f'{phase} Loss: {epoch_loss:.4f} AUC: {epoch_auc:.4f} Gamma: {epoch_gamma:.4f}') 

            if phase == 'train':
                gc.collect()         

            if (phase == 'val') and (epoch_auc > best_auc):

                param_name = os.path.join(model_save_path, f'{net.__class__.__name__.lower()}_{now.strftime("%Y%m%d%H%M%S")}_epoch_{epoch+1}_auc_{epoch_auc:.4f}.pth')

                best_auc = epoch_auc
                best_auc_model_path = param_name

                torch.save(net.state_dict(), param_name)
                logger.debug(f"New best model saved at {param_name}")
    
    logger.debug(f"Training complete. Best AUC: {best_auc:.4f}")
    
    return best_auc_model_path, loss_history, auc_history