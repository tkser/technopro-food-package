import json
import argparse

from typing import Optional

from models.PackageModel.train import train as train_PackageModel
from models.ResNet152Model.train import train as train_ResNet152Model
from models.ViTL16Model.train import train as train_ViTL16Model
from models.EfficientNetV2Model.train import train as train_EfficientNetV2Model

from utils.logger import logger


def train(model_name: str, batch_size: int, epochs: int, lr: float, seed: int, history_file: Optional[str] = None) -> str:
    if model_name == "package":
        model_path, loss_history, auc_history = train_PackageModel(batch_size, epochs, lr, seed)
    elif model_name == "resnet152":
        model_path, loss_history, auc_history = train_ResNet152Model(batch_size, epochs, lr, seed)
    elif model_name == "vitl16":
        model_path, loss_history, auc_history = train_ViTL16Model(batch_size, epochs, lr, seed)
    elif model_name == "efficientnetv2":
        model_path, loss_history, auc_history = train_EfficientNetV2Model(batch_size, epochs, lr, seed)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    if history_file is not None:
        with open(history_file, "w") as f:
            f.write(json.dumps({
                "loss": loss_history,
                "auc": auc_history
            }))
    
    return model_path


def main():
    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=16, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--history-file', type=Optional[str], default=None, help='Path to history file')

    args = parser.parse_args()

    logger.debug("Train model with following parameters:")
    logger.debug(f"Model: {args.model}")
    logger.debug(f"Batch size: {args.batch_size}")
    logger.debug(f"Epochs: {args.epochs}")
    logger.debug(f"Learning rate: {args.lr}")
    logger.debug(f"Random seed: {args.seed}")
    logger.debug(f"History file: {args.history_file}")

    bast_model_path = train(args.model, args.batch_size, args.epochs, args.lr, args.seed, args.history_file)
    logger.debug(f"Best model path: {bast_model_path}")


if __name__ == '__main__':
    main()