import os
import cv2
from torch.utils.data import Dataset


class ViTH14Dataset(Dataset):
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
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform[self.phase](image=image)["image"]
        
        label = self.label_list[idx]
        
        return image, label