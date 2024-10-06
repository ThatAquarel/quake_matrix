import torch

import numpy as np
import matplotlib.pyplot as plt

import d_vae_train as d


def eval(model, latent_dim, dataset):
    model.eval()

    with torch.no_grad():
        z = torch.randn(10, latent_dim)
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


N_GEN = 10000
GENERATED = "./dataset/QuakeVAE_generated_lunar_earthquakes.npz"


def main():
    dataset = d.QuakeDatasetVAE(lunar=True, debug=False)
    model = d.QuakeVAE(dataset.get_window_size_flat(), d.LATENT_DIM)

    state_dict = torch.load(d.QUAKE_VAE)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        z = torch.randn(N_GEN, d.LATENT_DIM)
        sxx = model.decode(z)
        sxx = sxx.reshape((N_GEN, *d.WINDOW_SIZE)).cpu()

        t, f = dataset.get_t(0).cpu(), dataset.get_f(0).cpu()

    plt.figure(figsize=(10, 5))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.pcolormesh(t, f, sxx[i])
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        plt.title(f"Generated {i+1}")

    plt.tight_layout()
    plt.savefig("./dataset/lunar_generated.png")
    plt.show()

    np.savez(GENERATED, sxx=sxx)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")

    main()
