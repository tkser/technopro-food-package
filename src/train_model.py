import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.set_seed import set_seed
from classes.PackageDataset import PackageDataset
from classes.PackageNet import PackageNet


def train(batch_size = 32, learning_rate = 0.001, num_epochs = 10):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_file_path = os.path.join(os.path.dirname(__file__), './data/input/train.csv')
    train_img_fd_path = os.path.join(os.path.dirname(__file__), './data/input/train')

    train_dataset = PackageDataset(csv_file=train_file_path, root_dir=train_img_fd_path, transform=transform, type='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = PackageNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    test_file_path = os.path.join(os.path.dirname(__file__), './data/input/sample_submit.csv')
    test_img_fd_path = os.path.join(os.path.dirname(__file__), './data/input/test')
    
    test_dataset = PackageDataset(csv_file=test_file_path, root_dir=test_img_fd_path, transform=transform, type='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            outputs = model(images)
            predicts = outputs.softmax(dim=1)
            predictions.append(predicts)
    
    predictions = torch.concatenate(predictions, dim=0).cpu().numpy()
    predictions = predictions[:, 1]
    
    output_file_path = os.path.join(os.path.dirname(__file__), './data/output/predictions.csv')
    
    submission_df = pd.read_csv(test_file_path, header=None, names=['img_name', 'label'])
    submission_df['label'] = predictions
    submission_df.to_csv(output_file_path, index=False, header=False)


if __name__ == '__main__':
    train()