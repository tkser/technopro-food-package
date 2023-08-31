import os
import torch
import logging
from sklearn.metrics import roc_auc_score

from typing import Callable, Optional


def predict(
        net: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: Optional[str] = None
    ) -> list :

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net.to(device)

    pred_list = []

    for images, _ in dataloader:

        images = images.float().to(device)

        outputs = net(images)

        _, preds = torch.max(outputs, 1)
        preds = preds.to('cpu').numpy()

        pred_list.extend(preds)
    
    return pred_list