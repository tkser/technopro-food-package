from models.PackageModel.train import train as train_PackageModel

from utils.logger import logger


def train():
    package_model_path, _, _ = train_PackageModel()
    logger.debug(f"PackageModel: {package_model_path}")


if __name__ == '__main__':
    train()