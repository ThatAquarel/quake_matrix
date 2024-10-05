import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class SpectogramCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpectogramCNN, self).__init__()
        self.first_conv_layer = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=4, stride=1, padding=1)
        self.pool_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.second_conv_layer = nn.Conv2d(20, 40, kernel_size=4, stride=1, padding=1)
        self.third_conv_layer = nn.Conv2d(40, 80, kernel_size=4, stride=1, padding=1)
        self.first_connect_layer = nn.Linear(80 * 30 * 30, out_features=256)
        self.second_connect_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool_layer(F.relu(self.first_conv_layer(x)))
        x = self.pool_layer(F.relu(self.second_conv_layer(x)))
        x = self.pool_layer(F.relu(self.third_conv_layer(x)))
        x = x.view(-1, 80 * 30 * 30)
        x = F.relu(self.first_connect_layer(x))
        x = self.second_connect_layer(x)
        return x


def objective(trial):
    num_classes = 2
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    num_epochs = trial.suggest_int('num_epochs', 5, 50)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = ImageFolder('pathdata', transform=transform)
    train_data, val_data = train_test_split(dataset, test_size=0.2)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = SpectogramCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(study.best_params)
