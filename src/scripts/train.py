import os
import logging

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from typing import Dict, Callable, Optional


def train(
        net: nn.Module,
        epochs: int,
        loss_fn: Callable,
        optimizer: optim.Optimizer,
        dataloaders: Dict[str, DataLoader],
        device: Optional[str] = None,
        model_save_path: Optional[str] = None
    ) -> Tuple[Optional[str], Dict[str, list], Dict[str, list]] :

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net.to(device)

    best_auc = 0.0
    best_auc_model_path = None

    loss_history = {'train': [], 'val': []}
    auc_history = {'train': [], 'val': []}

    for epoch in range(epochs):

        logging.debug(f'Epoch: {epoch+1} / {epochs}')
        logging.debug('--------------------------')

        for phase in ['train', 'val']:

            if phase == 'train':
                net.train()
            else:
                net.eval()
        
            epoch_loss = 0.0
            pred_list = []
            true_list = []

            for images, labels in dataloaders[phase]:

                images = images.float().to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):

                    outputs = net(images)
                    loss = loss_fn(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    epoch_loss += loss.item() * images.size(0)

                    preds = preds.to('cpu').numpy()
                    pred_list.extend(preds)

                    labels = labels.to('cpu').numpy()
                    true_list.extend(labels)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

            epoch_auc = roc_auc_score(true_list, pred_list)

            loss_history[phase].append(epoch_loss)
            auc_history[phase].append(epoch_auc)

            logging.debug(f'{phase} Loss: {epoch_loss:.4f} AUC: {epoch_auc:.4f}')            

            if (phase == 'val') and (epoch_auc > best_auc):

                param_name = os.path.join(model_save_path, f'Epoch{epoch+1}_auc_{epoch_auc:.4f}.pth')

                best_auc = epoch_auc
                best_auc_model_path = param_name

                torch.save(net.state_dict(), param_name)
    
    return best_auc_model_path, loss_history, auc_history