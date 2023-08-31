import torch

from models.PackageModel.Net import PackageNet
from models.PackageModel.Dataset import PackageDataset

from scripts.predict import predict as predict_model
from utils.set_seed import set_seed


def predict(model_path: str, batch_size = 32, seed = 42):

    set_seed(seed)

    transform = {
        "train": transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]),
        "val": transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]),
    }

    test_file_path = os.path.join(os.path.dirname(__file__), '../../data/input/sample_submit.csv')
    test_img_fd_path = os.path.join(os.path.dirname(__file__), '../../data/input/images/test')
    submission_file_path = os.path.join(os.path.dirname(__file__), '../../data/output/submit.csv')

    sample_submission = pd.read_csv(test_file_path, header=None, names=['img_name', 'label'])

    X_test = sample_submission['image_name'].values
    dummy = sample_submission['label'].values
    
    test_dataset = PackageDataset(X_test, dummy, root_dir=test_img_fd_path, transform=transform, phase='val')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = PackageNet()

    trained_params = torch.load(model_path)
    model.load_state_dict(trained_params)

    y_pred = predict_model(model, test_loader)

    sample_submission["label"] = y_pred
    sample_submission.to_csv(submission_file_path, index=False, header=False)

    return submission_file_path, y_pred