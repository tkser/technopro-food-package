import logging

from models.PackageModel.train import train as train_PackageModel


def train():
    package_model_path, _, _ = train_PackageModel()
    logging.debug(f"PackageModel: {package_model_path}")


if __name__ == '__main__':
    train()