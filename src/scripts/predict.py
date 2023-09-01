import gc
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.logger import logger

from typing import Optional


def predict(
        net: torch.nn.Module,
        dataloader: DataLoader,
        device: Optional[str] = None
    ) -> list :

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net.to(device)

    logger.debug(f"Starting prediction on {device}")

    pred_list = []

    with torch.no_grad():
        for images, _ in  tqdm(dataloader, desc=f"Prediction"):

            images = images.float().to(device)

            outputs = net(images)

            preds = torch.softmax(outputs, dim=1)
            
            preds = preds.to('cpu').numpy()

            pred_list.append(preds)

            del images, outputs, preds
            torch.cuda.empty_cache()
    
    pred_list = np.concatenate(pred_list, axis=0)
    pred_list = pred_list[:, 1]

    gc.collect()
    
    logger.debug(f"Finished prediction")
    
    return pred_list