import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import optuna
import numpy as np

import d_vae_train as d
import e_vae_infer as e


class QuakeDatasetCNN(d.QuakeDatasetVAE):
    def __init__(self, lunar=True, train=True):
        super().__init__()

        self.train = train
        self.lunar = lunar

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if not self.lunar:
            raise NotImplementedError()

        if self.train:
            self.load_db_train(e.GENERATED)

    def load_db_train(self, file):
        vae_data = np.load(file)
        sxx = vae_data["sxx"]

        window_size = self.get_window_size()
        self.gen_quake = torch.from_numpy(sxx).reshape((-1, 1, *window_size))
        self.no_quake = torch.zeros((1, *window_size))
        self.rand = torch.randint(0, 2, (len(self),))

        self.gen_quake = self.gen_quake.to(self.device)
        self.no_quake = self.no_quake.to(self.device)

    def __len__(self):
        if self.train:
            return self.gen_quake.shape[0]
        return super().__len__()

    def __getitem__(self, idx):
        if self.train:
            if self.rand[idx]:
                return self.gen_quake[idx], 1
            return (self.no_quake, 0)

        sxx = super().__getitem__(idx)
        return sxx.reshape((1, *self.get_window_size())), 1


class QuakeCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(QuakeCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(384 * 3 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.cnn(x)


def objective(trial, dataset_train, dataset_test):
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    num_epochs = trial.suggest_int("num_epochs", 5, 50)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size)

    model = QuakeCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in dataloader_train:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            print(f"epoch {epoch}, Loss: {loss.item()}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader_test:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model_file = f"./model/QuakeCNN_{trial.number}.pth"
    torch.save(model.state_dict(), model_file)
    trial.set_user_attr("model_file", model_file)

    accuracy = correct / total
    return accuracy


def main(
    n=4,
):
    dataset_train = QuakeDatasetCNN(train=True)
    dataset_test = QuakeDatasetCNN(train=False)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, dataset_train, dataset_test), n_trials=n
    )

    df = study.trials_dataframe()
    df.to_csv("./model/bayesian_optimization.csv", index=False)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")

    main()
