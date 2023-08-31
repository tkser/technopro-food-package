from models.PackageModel.predict import predict as predict_PackageModel


def predict():
    predict_PackageModel("./src/data/models/PackageModel/Epoch1_auc_0.000.pth")


if __name__ == "__main__":
    predict()