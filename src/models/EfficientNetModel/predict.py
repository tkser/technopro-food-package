import os
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.pytorch as APT
from torchvision.models import efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6
from torchvision.models import EfficientNet_B2_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights

from models.EfficientNetModel.Dataset import EfficientNetDataset

from scripts.predict import predict as predict_model
from utils.set_seed import set_seed


def predict(model_path: str, batch_size = 32, seed = 42, model_type = 'b3'):

    set_seed(seed)

    image_size = 0
    if model_type == 'b2':
        image_size = 288
    elif model_type == 'b3':
        image_size = 320
    elif model_type == 'b4':
        image_size = 384
    elif model_type == 'b5':
        image_size = 456
    elif model_type == 'b6':
        image_size = 528
    
    if image_size == 0:
        raise ValueError('Invalid model_type')

    transform = {
        "train": A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(),
            APT.ToTensorV2(),
        ]),
        "val": A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(),
            APT.ToTensorV2()
        ]),
    }

    test_file_path = os.path.join(os.path.dirname(__file__), '../../data/input/sample_submit.csv')
    test_img_fd_path = os.path.join(os.path.dirname(__file__), '../../data/input/images/test')

    model_path = model_path.replace('\\', '/')

    submission_file_name = f"submit_{'.'.join(model_path.split('/')[-1].split('.')[0:-1])}.csv"
    submission_file_path = os.path.join(os.path.dirname(__file__), '../../data/output', submission_file_name)

    sample_submission = pd.read_csv(test_file_path, header=None, names=['image_name', 'label'])

    X_test = sample_submission['image_name'].values
    dummy = sample_submission['label'].values
    
    test_dataset = EfficientNetDataset(X_test, dummy, root_dir=test_img_fd_path, transform=transform, phase='val')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = None

    if model_type == 'b2':
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    elif model_type == 'b3':
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    elif model_type == 'b4':
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    elif model_type == 'b5':
        model = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
    elif model_type == 'b6':
        model = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
    
    if model is None:
        raise ValueError('Invalid model_type')

    model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=2)

    trained_params = torch.load(model_path)
    model.load_state_dict(trained_params)

    y_pred = predict_model(model, test_loader)

    sample_submission["label"] = y_pred
    sample_submission.to_csv(submission_file_path, index=False, header=False)

    return submission_file_path, y_pred