import os
import scipy

import numpy as np
import pandas as pd
import scipy.signal

from tqdm import tqdm
from pathlib import Path


def recursive_files(dir, ext_filter=".csv"):
    for i in os.listdir(dir):
        current_path = os.path.join(dir, i)
        if os.path.isdir(current_path):
            for j in recursive_files(current_path):
                yield j

        _, extension = os.path.splitext(i)
        if extension == ext_filter:
            yield current_path


def get_sampling_rate(series):
    return 1 / np.mean(series[1:] - series[:-1])


def data_parse(csv_data):
    try:
        # lunar data headers
        t = np.array(csv_data["time_rel(sec)"])
        v = np.array(csv_data["velocity(m/s)"])
        default_sr = 6.625  # Hz
    except KeyError:
        # mars data headers
        t = np.array(csv_data["rel_time(sec)"])
        v = np.array(csv_data["velocity(c/s)"])
        default_sr = 20.00  # Hz

    sr = get_sampling_rate(t)
    if not np.isclose(sr, default_sr):
        raise ValueError(f"sr {sr} mistmatch with default {default_sr}")

    return t, v, sr


def process(
    data_dir,
    preprocess_dir,
    lower_freq=0.0,
    upper_freq=1.5,
):
    for file_dir in tqdm(recursive_files(data_dir)):
        try:
            csv_data = pd.read_csv(file_dir)
        except FileNotFoundError:
            print(f"skipped {file_dir}")
            continue

        _, v, sr = data_parse(csv_data)
        spec_f, spec_t, sxx = scipy.signal.spectrogram(v, sr)

        mask = (spec_f > lower_freq) & (spec_f < upper_freq)
        spec_f, sxx = spec_f[mask], sxx[mask, :]

        rel_path = os.path.relpath(file_dir, data_dir)
        save_file = os.path.join(preprocess_dir, rel_path)
        save_path = os.path.dirname(save_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.savez(f"{save_file}.npz", spec_f, spec_t, sxx)


def main(
    data_dir="./data/data/",
    preprocess_dir="./dataset/data/",
):
    for planet in ["mars/", "lunar/"]:
        for dataset in ["test/", "training/"]:
            process(
                f"{data_dir}{planet}{dataset}data/",
                f"{preprocess_dir}{planet}{dataset}",
            )


if __name__ == "__main__":
    main()
