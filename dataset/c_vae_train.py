import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as nn_data

import numpy as np
import pandas as pd

import a_generate_spectrogram as a


class QuakeDataset(nn_data.Dataset):
    def __init__(self, train=True, lunar=True) -> None:
        super().__init__()

        self.train = train
        self.lunar = lunar

        self.load_catalog()
        self.load_db()

    def load_catalog(
        self,
        catalog_lunar="lunar/training/catalogs/apollo12_catalog_GradeA_final.csv",
        catalog_mars="mars/training/catalogs/Mars_InSight_training_catalog_final.csv",
    ):
        if self.lunar:
            catalog_dir = f"{a.DATA_DIR}{catalog_lunar}"
        else:
            catalog_dir = f"{a.DATA_DIR}{catalog_mars}"

        catalog_data = pd.read_csv(catalog_dir)

        pass

    def load_db(self):
        if self.train:
            self.x = torch.load(f"./prediction/x_2003-2017_train.pt")
            self.y = torch.load(f"./prediction/y_2003-2017_train.pt")
            return

        self.x = torch.load(f"./prediction/x_2018_test.pt")
        self.y = torch.load(f"./prediction/y_2018_test.pt")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


class QuakeVAE(nn.Module):
    def __init__(self):
        super(QuakeVAE, self).__init__()

        # encoder layers
        self.f1_enc = nn.Linear(input_dim, 50)
        self.f2_mu = nn.Linear(50, latent_dim)
        self.f2_logvar = nn.Linear(50, latent_dim)

        # decoder layers
        self.fc3 = nn.Linear(latent_dim, 50)
        self.fc4 = nn.Linear(50, input_dim)

    def encode(self, x):
        # encode data into latent space
        h1 = F.relu(self.f1_enc(x))
        return self.f2_mu(h1), self.f2_logvar(h1)

    def reparameterize(self, mu, logvar):
        # random sample in latent space distribution
        stdev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stdev)
        return mu + epsilon * stdev

    def decode(self, z):
        # decode data from latent spaces
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(x_gen, x, mu, logvar):
    mse = F.mse_loss(x_gen, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld


def main(
    lr=1e-3,
    epochs=1000,
    batch_size=32,
):
    model = QuakeVAE()
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x_gen, mu, logvar = model(x)
        loss = loss_function(x_gen, x, mu, logvar)

        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
