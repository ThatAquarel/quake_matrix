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
        sxx = self.get_sxx(idx)
        return sxx.reshape((1, *sxx.shape))


class QuakeVAE(nn.Module):
    def __init__(self, window_size, latent_dim=4):
        super(QuakeVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=4, stride=2, padding=1
            ),  # (B, 32, img_size/2, img_size/2)
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=4, stride=2, padding=1
            ),  # (B, 64, img_size/4, img_size/4)
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),  # (B, 128, img_size/8, img_size/8)
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the output size after convolutions for fully connected layers
        conv_img_size = np.array(window_size) // 8
        conv_output_size = np.multiply.reduce(conv_img_size) * 128

        # Latent space representation
        self.fc_mu = nn.Linear(conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(conv_output_size, latent_dim)

        # Fully connected layer to map latent space to decoded output
        self.fc_decode = nn.Linear(latent_dim, conv_output_size)

        # Define the Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, int(conv_img_size[0]), int(conv_img_size[1]))),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # (B, 64, img_size/4, img_size/4)
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # (B, 32, img_size/2, img_size/2)
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, kernel_size=4, stride=2, padding=1
            ),  # (B, 1, img_size, img_size)
            nn.Sigmoid(),  # Output in range [0, 1] for normalized images
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Reparameterization trick

    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def loss_function(x_gen, x, mu, logvar):
    mse = F.mse_loss(x_gen, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld


def main(
    lr=1e-3,
    epochs=256,
    batch_size=4,
):
    dataset = QuakeDatasetVAE(lunar=True, debug=False)
    dataloader = nn_data.DataLoader(dataset, batch_size=batch_size)

    model = QuakeVAE(dataset.get_window_size())
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
        z = torch.randn(10, 4)
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
