import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as nn_data

import numpy as np
import pandas as pd

import a_generate_spectrogram as a
import b_verify_spectrogram as b
import c_generate_catalog as c


class QuakeDatasetVAE(nn_data.Dataset):
    def __init__(self, lunar=True, debug=False) -> None:
        super().__init__()
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
        self.indices = self.indices.type(torch.int32)

    def verify_indices(self):
        for i in range(len(self)):
            t = self.get_t(i)
            f = self.get_f(i)
            sxx = self.get_sxx(i)

            b.spectrogram(t, f, sxx)

    def get_t(self, idx):
        return self.t_all[idx, self.indices[idx]]

    def get_f(self, idx):
        return self.f_all[idx]

    def get_sxx(self, idx):
        return self.sxx_all[idx, :, self.indices[idx]]

    def get_window_size(self):
        return self.f_all.shape[1], self.indices.shape[1]

    def get_window_size_flat(self):
        return np.multiply.reduce(self.get_window_size())

    def __len__(self):
        return self.indices.size(0)

    def __getitem__(self, idx):
        return self.get_sxx(idx).flatten()


class QuakeVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, inner_enc):
        super(QuakeVAE, self).__init__()

        # encoder layers
        self.f1_enc = nn.Linear(input_dim, inner_enc)
        self.f2_mu = nn.Linear(inner_enc, latent_dim)
        self.f2_logvar = nn.Linear(inner_enc, latent_dim)

        # decoder layers
        self.fc3 = nn.Linear(latent_dim, inner_enc)
        self.fc4 = nn.Linear(inner_enc, input_dim)

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
        # decode data from latent space
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
    lr=1e-4,
    epochs=1000,
    batch_size=32,
):
    dataset = QuakeDatasetVAE(lunar=True, debug=False)
    dataloader = nn_data.DataLoader(dataset, batch_size=batch_size)

    model = QuakeVAE(dataset.get_window_size_flat(), 512, 1024)
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(epochs):
        model.train()
        for sxx in dataloader:
            sxx_gen, mu, logvar = model(sxx)
            loss = loss_function(sxx_gen, sxx, mu, logvar)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            print(f"epoch {epoch}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        z = torch.randn(10, 512)
        sxx = model.decode(z)
        sxx = sxx.reshape((10, *dataset.get_window_size())).cpu()

        t, f = dataset.get_t(0).cpu(), dataset.get_f(0).cpu()
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.pcolormesh(t, f, sxx[i])
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Time [sec]")
            plt.title(f"Generated {i+1}")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")

    main()
