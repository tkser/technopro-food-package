import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PackageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label