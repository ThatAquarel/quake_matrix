import os

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt


import a_generate_spectrogram as a


QUAKE_THRESHOLD = 1e-13


def get_quakes(arr):
    # Check if the array is empty
    if arr.size == 0:
        return arr

    # Create a mask for the first element of each sequence
    mask = np.abs(arr[1:] - arr[:-1]) > 16
    mask = np.concatenate(([True], mask))

    seq_start = arr[mask]
    mean_seq_len = len(mask) // len(seq_start)

    adj_positions = seq_start - mean_seq_len

    return np.clip(adj_positions, 0, adj_positions.max())


def main():
    softmax = nn.Softmax(dim=1)

    time_rel = []
    filename = []

    for file in a.recursive_files(a.PREPROCESS_DIR, ext_filter=".pth"):
        if "mars" in file:
            continue

        d = np.load(file[:-4])
        f, t, sxx = d["spec_f"], d["spec_t"], d["sxx"]
        t -= t.min()

        y = torch.load(file)
        prob = softmax(y)

        plt.figure(figsize=(12, 6))
        ax0 = plt.subplot(3, 1, 1)
        ax0.pcolormesh(t, f, sxx, cmap="plasma")
        ax0.set_xlim([t.min(), t.max()])
        ax0.set_ylabel("Frequency [Hz]")

        ax1 = plt.subplot(3, 1, 2)
        prob_quake = prob[:, 1].numpy()
        prob_t = t[: prob_quake.size]
        ax1.plot(prob_t, prob_quake)
        ax1.set_xlim([t.min(), t.max()])
        ax1.set_ylabel("Predicted quake probability")
        ax1.set_xlabel("Time [sec]")

        dy = np.diff(prob[:, 1])
        (indices,) = np.where(dy > QUAKE_THRESHOLD)
        quakes = get_quakes(indices)

        for quake in quakes:
            quake_t = t[quake + 64]
            ax0.axvline(x=quake_t, c="red", label="Abs. Arrival")
            ax1.axvline(x=quake_t, c="red", label="Abs. Arrival")

        plt.show()


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    main()
