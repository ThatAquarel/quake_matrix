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
    def __init__(self, train=True, lunar=True, debug=False) -> None:
        super().__init__()

        self.train = train
        self.lunar = lunar

        self.load_catalog()
        self.compute_index()

        if debug:
            self.verify_indices()

    def load_catalog(self):
        if self.lunar:
            catalog_dir = c.CATALOG_LUNAR
            file_dir = f"{a.PREPROCESS_DIR}lunar/training/S12_GradeA/"
        else:
            catalog_dir = c.CATALOG_MARS
            file_dir = f"{a.PREPROCESS_DIR}mars/training/"

        catalog_data = pd.read_csv(catalog_dir)

        allocated = False
        all_len = len(catalog_data)

        for i, row in catalog_data.iterrows():
            filename = f"{row['filename']}.npz"

            d = np.load(f"{file_dir}{filename}")
            f, t, sxx = d["spec_f"], d["spec_t"], d["sxx"]

            if not allocated:
                self.f_all = torch.empty((all_len, *f.shape))
                self.t_all = torch.empty((all_len, *t.shape))
                self.t_size = torch.empty(all_len)
                self.sxx_all = torch.empty((all_len, *sxx.shape))

                self.time_rel = torch.empty(all_len)
                allocated = True

            self.f_all[i, : f.size] = torch.from_numpy(f)
            self.t_all[i, : t.size] = torch.from_numpy(t)
            self.t_size[i] = t.size
            self.sxx_all[i, : f.size, : t.size] = torch.from_numpy(sxx)

            self.time_rel[i] = row["time_rel(sec)"]

    def compute_index(self, pre=4, post=124):
        dt = torch.abs(self.t_all - self.time_rel.view(-1, 1))

        n = dt.shape[0]

        index_center = torch.empty((n, 1))
        for i, series in enumerate(dt):
            t_size = int(self.t_size[i])
            index_center[i, 0] = torch.argmin(series[:t_size])

        window_size = pre + post
        windows = torch.tile(torch.arange(window_size), (n, 1))
        window_index = index_center - pre + windows

        self.indices = window_index % self.t_size.view(-1, 1)

    def verify_indices(self):
        import matplotlib.pyplot as plt

        for i, indices in enumerate(self.indices):
            indices = indices.type(torch.int32)

            t = self.t_all[i, indices]
            f = self.f_all[i]
            sxx = self.sxx_all[i, :, indices]

            plt.pcolormesh(t, f, sxx)
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Time [sec]")
            plt.show()

    def __len__(self):
        return self.indices.size

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

    train_dataloader = nn_data.DataLoader(
        QuakeDatasetVAE(debug=True), batch_size=batch_size
    )

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x_gen, mu, logvar = model(x)
        loss = loss_function(x_gen, x, mu, logvar)

        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
