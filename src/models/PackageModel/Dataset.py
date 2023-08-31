import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PackageDataset(Dataset):
    def __init__(self, image_name_list, label_list, root_dir, transform=None, phase='train'):
        self.image_name_list = image_name_list
        self.label_list = label_list
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_name_list[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform[self.phase](image)
        
        label = self.label_list[idx]
        
        return image, label