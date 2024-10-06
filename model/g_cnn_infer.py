import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import d_vae_train as d
import f_cnn_train as cnn


def load_best_candidate(model):
    bayesian_results = pd.read_csv(cnn.BAYESIAN_RESULTS)
    best_candidate = bayesian_results.iloc[bayesian_results["value"].idxmax()]

    model_file = best_candidate["user_attrs_model_file"]
    state_dict = torch.load(model_file)

    model.load_state_dict(state_dict)


def main():
    vae_dataset = d.QuakeDatasetVAE(lunar=True, debug=False)
    cnn_dataset = cnn.QuakeDatasetCNN(lunar=True, train=False)

    model = cnn.QuakeCNN()
    load_best_candidate(model)
    model.eval()

    softmax = nn.Softmax(dim=1)

    plt.figure(figsize=(12, 6))

    for i in range(8):
        if i % 2 == 0:
            (x, _) = cnn_dataset[i]
            sxx = x.reshape((1, 1, 64, 128))
        else:
            sxx = torch.zeros((1, 1, 64, 128))

        y = model(sxx)
        prob = softmax(y)
        prob = prob.reshape(-1)
        quake_conf = float(prob[1])
        no_quake_conf = float(prob[0])

        t, f = vae_dataset.get_t(0).cpu(), vae_dataset.get_f(0).cpu()
        t -= t.min()

        plt.subplot(2, 4, i + 1)
        plt.pcolormesh(t, f, sxx.reshape((64, 128)), cmap="plasma")
        plt.ylabel("Frequency [Hz]", fontsize=6)
        plt.xlabel("Time [sec]", fontsize=6)
        plt.tick_params(axis="both", which="major", labelsize=4)

        if quake_conf > no_quake_conf:
            quake = " \\textbf{Quake}"
            no_quake = "No quake"
        else:
            quake = "Quake"
            no_quake = "\\textbf{No quake}"

        plt.title(
            f"{quake}, confidence: {quake_conf:.3g}\n"
            f"{no_quake}, confidence: {no_quake_conf:.3g}",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig("./model/g_cnn_infer_predicted_lunar.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    main()
