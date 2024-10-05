import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as nn_data

import numpy as np
import pandas as pd

import a_generate_spectrogram as a
import c_generate_catalog as c


class QuakeDatasetVAE(nn_data.Dataset):
    def __init__(self, train=True, lunar=True) -> None:
        super().__init__()

        self.train = train
        self.lunar = lunar

        self.load_catalog()
        self.load_db()

    def load_catalog(
        self,
    ):
        if self.lunar:
            catalog_dir = c.CATALOG_LUNAR
            file_dir = f"{a.PREPROCESS_DIR}lunar/training/S12_GradeA/"
            file_ext = ".csv.npz"
        else:
            catalog_dir = c.CATALOG_MARS
            file_dir = f"{a.PREPROCESS_DIR}mars/training/"
            file_ext = ".npz"

        catalog_data = pd.read_csv(catalog_dir)

        for _, row in catalog_data.iterrows():
            filename = f"{row['filename']}{file_ext}"
            time_rel = row["time_rel(sec)"]

            d = np.load(f"{file_dir}{filename}")
            spec_f, spec_t, sxx = d["spec_f"], d["spec_t"], d["sxx"]

        ...

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


input_dim = 100
latent_dim = 2


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

    train_dataloader = nn_data.DataLoader(QuakeDatasetVAE(), batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x_gen, mu, logvar = model(x)
        loss = loss_function(x_gen, x, mu, logvar)

        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
