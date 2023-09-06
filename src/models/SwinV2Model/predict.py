import os
import timm
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.pytorch as APT

from models.SwinV2Model.Dataset import SwinV2Dataset

from scripts.predict import predict as predict_model
from utils.set_seed import set_seed


def predict(model_path: str, batch_size = 32, seed = 42):

    set_seed(seed)

    transform = {
        "train": A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            APT.ToTensorV2(),
        ]),
        "val": A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            APT.ToTensorV2()
        ]),
    }

    test_file_path = os.path.join(os.path.dirname(__file__), '../../data/input/sample_submit.csv')
    test_img_fd_path = os.path.join(os.path.dirname(__file__), '../../data/input/images/test')

    submission_file_name = f"submit_{'.'.join(model_path.split('/')[-1].split('.')[0:-1])}.csv"
    submission_file_path = os.path.join(os.path.dirname(__file__), '../../data/output', submission_file_name)

    sample_submission = pd.read_csv(test_file_path, header=None, names=['image_name', 'label'])

    X_test = sample_submission['image_name'].values
    dummy = sample_submission['label'].values
    
    test_dataset = SwinV2Dataset(X_test, dummy, root_dir=test_img_fd_path, transform=transform, phase='val')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = timm.create_model('swinv2_large_window12to24_192to384', pretrained=True, num_classes=2)

    trained_params = torch.load(model_path)
    model.load_state_dict(trained_params)

    y_pred = predict_model(model, test_loader)

    sample_submission["label"] = y_pred
    sample_submission.to_csv(submission_file_path, index=False, header=False)

    return submission_file_path, y_pred