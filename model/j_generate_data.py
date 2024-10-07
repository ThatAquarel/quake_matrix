import os

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
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

    filenames = []
    time_rels = []

    js_data = open("./docs/data.js", "w+")
    js_data.write("data=[\n")

    for file in a.recursive_files(a.PREPROCESS_DIR, ext_filter=".pth"):
        skip = True
        for string in [
            "00006",
            "00007",
            "00018",
            "00021",
            "00026",
            "00028",
            "00031",
            "00032",
            "00033",
            "00039",
        ]:
            if string in file:
                skip = False
                break
        if skip:
            continue

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

        filename = os.path.basename(file)
        quake_t = None
        for quake in quakes:
            quake_t = t[quake + 64]

            filenames.append(filename[:-8])
            time_rels.append(quake_t)

            quake_t = quake

            ax0.axvline(x=quake_t, c="red", label="Abs. Arrival")
            ax1.axvline(x=quake_t, c="red", label="Abs. Arrival")

        time_data = pd.read_csv(
            file[:-8].replace("dataset", "data").replace("test\\", "test\\data\\")
        )
        vel = time_data[["velocity(m/s)"]]
        scale_factor = len(vel) // len(t)

        averaged = vel.groupby(np.arange(len(vel)) // scale_factor).mean()
        averaged = np.array(averaged).reshape(-1)

        principal_f = np.argmax(sxx, axis=0)
        selected_f = f[principal_f]

        sample_name = filename[14:-12]

        def get_str(arr):
            builder = "["
            for i in arr:
                builder += f"{i:.5f},"
            builder += "]"
            return builder

        def normalize(arr):
            arr_min = np.min(arr)
            arr_max = np.max(arr)
            return (arr - arr_min) / (arr_max - arr_min)

        t_str = get_str(prob_t)
        p_str = get_str(normalize(prob_quake))
        f_str = get_str(selected_f)
        a_str = get_str(normalize(averaged))

        js_data.write("{\n")
        js_data.write(f'name: "{sample_name}",\n')
        js_data.write(f"t: {t_str},\n")
        js_data.write(f"p: {p_str},\n")
        js_data.write(f"f: {f_str},\n")
        js_data.write(f"a: {a_str},\n")
        js_data.write("},\n")

    js_data.write("];\n")
    js_data.close()


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    main()
