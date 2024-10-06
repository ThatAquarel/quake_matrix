import torch

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import a_generate_spectrogram as a
import d_vae_train as vae
import f_cnn_train as cnn
import g_cnn_infer as g


class QuakeFullDatasetCNN(vae.QuakeDatasetVAE):
    def __init__(self, lunar=True) -> None:
        self.lunar = lunar

        super().__init__()

        self.compute_mapping()
        self.load_all()

    def compute_mapping(self):
        min_0, max_0 = self.sxx_all_before_norm.min(), self.sxx_all_before_norm.max()
        min_1, max_1 = self.sxx_all.min(), self.sxx_all.max()

        self.remap = lambda x: min_1 + (x - min_0) * (max_1 - min_1) / (max_0 - min_0)

    def load_all(self):
        self.batches = []

        for file in a.recursive_files(a.PREPROCESS_DIR, ext_filter=".npz"):
            if "mars" in file:
                continue

            d = np.load(file)
            f, t, sxx = d["spec_f"], d["spec_t"], torch.from_numpy(d["sxx"]).to("cuda")
            sxx = self.remap(sxx)

            n_views = len(t) - 128 + 1
            batch = torch.empty((n_views, 1, 64, 128))
            for i in range(n_views):
                batch[i, :] = sxx[:, i : i + 128]

            self.batches.append((file, f, t, batch))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


def main():
    model = cnn.QuakeCNN()
    g.load_best_candidate(model)
    model.eval()

    dataset = QuakeFullDatasetCNN()
    dataset = list(dataset)

    for file, _, _, batch in tqdm(dataset):
        y = model(batch)
        torch.save(y.cpu(), f"{file}.pth")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")

    plt.rcParams["text.usetex"] = True
    main()
