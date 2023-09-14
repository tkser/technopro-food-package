import argparse

from models.PackageModel.predict import predict as predict_PackageModel
from models.ResNet152Model.predict import predict as predict_ResNet152Model
from models.ViTL16Model.predict import predict as predict_ViTL16Model
from models.ViTH14Model.predict import predict as predict_ViTH14Model
from models.EfficientNetV2Model.predict import predict as predict_EfficientNetV2Model
from models.SwinV2Model.predict import predict as predict_SwinV2Model
from models.ConvNextV2.predict import predict as predict_ConvNextV2Model

from utils.logger import logger


def predict(model_name: str, model_path: str, batch_size: int, seed: int, use_tta: bool = False) -> str:
    if model_name == "package":
        submission_file_path = predict_PackageModel(model_path, batch_size, seed)
    elif model_name == "resnet152":
        submission_file_path = predict_ResNet152Model(model_path, batch_size, seed)
    elif model_name == "vitl16":
        submission_file_path = predict_ViTL16Model(model_path, batch_size, seed)
    elif model_name == "vith14":
        submission_file_path = predict_ViTH14Model(model_path, batch_size, seed)
    elif model_name == "efficientnetv2":
        submission_file_path = predict_EfficientNetV2Model(model_path, batch_size, seed)
    elif model_name == "swinv2":
        submission_file_path = predict_SwinV2Model(model_path, batch_size, seed, use_tta=use_tta)
    elif model_name == "swinv1":
        submission_file_path = predict_SwinV2Model(model_path, batch_size, seed, model_name="swin_large_patch4_window12_384", use_tta=use_tta)
    elif model_name == "convnextv2":
        submission_file_path = predict_ConvNextV2Model(model_path, batch_size, seed)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return submission_file_path


def main():
    parser = argparse.ArgumentParser(description='Predict models')
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('--model-path', type=str, help='Path to model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-tta', type=bool, default=False, help='Use test time augmentation')

    args = parser.parse_args()

    logger.debug("Predict model with following parameters:")
    logger.debug(f"Model: {args.model}")
    logger.debug(f"Model path: {args.model_path}")
    logger.debug(f"Batch size: {args.batch_size}")
    logger.debug(f"Random seed: {args.seed}")
    logger.debug(f"Use test time augmentation: {args.use_tta}")

    submission_file_path = predict(args.model, args.model_path, args.batch_size, args.seed, args.use_tta)
    logger.debug(f"Submission file path: {submission_file_path}")


if __name__ == "__main__":
    main()