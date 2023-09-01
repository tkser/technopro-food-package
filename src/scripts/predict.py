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

    for images, _ in  tqdm(dataloader, desc=f"Prediction"):

        images = images.float().to(device)

        outputs = net(images)

        _, preds = torch.max(outputs, 1)
        preds = preds.to('cpu').numpy()

        pred_list.extend(preds)
    
    logger.debug(f"Finished prediction")
    
    return pred_list