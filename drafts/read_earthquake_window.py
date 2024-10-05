# Import libraries
import os

import numpy as np
import pandas as pd

# from obspy import read

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy import signal


CATALOG_FILE = "./data/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv"
DATA_DIR = "./data/data/lunar/training/data/S12_GradeA/"
catalog = pd.read_csv(CATALOG_FILE)

# PRE_WINDOW_SIZE = 10
# POST_WINDOW_SIZE =

SAMPLING_RATE = 6.625  # Hz


for i, event in catalog.iterrows():
    csv_file = f"{DATA_DIR}{event.filename}.csv"
    try:
        data_cat = pd.read_csv(csv_file)
    except FileNotFoundError:
        continue

    csv_times = np.array(data_cat["time_rel(sec)"].tolist())
    csv_data = np.array(data_cat["velocity(m/s)"].tolist())

    # Plot the trace!
    fig = plt.subplots(figsize=(10, 10))
    ax = plt.subplot(2, 1, 1)
    ax.plot(csv_times, csv_data)

    # Make the plot pretty
    ax.set_xlim([min(csv_times), max(csv_times)])
    ax.set_ylabel("Velocity (m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{event.filename}", fontweight="bold")

    # Plot where the arrival time is
    arrival_time_rel = event["time_rel(sec)"]

    arrival_line = ax.axvline(x=arrival_time_rel, c="red", label="Rel. Arrival")
    ax.legend(handles=[arrival_line])

    f, t, sxx = signal.spectrogram(csv_data, SAMPLING_RATE)

    # format
    (index,) = np.where(f < 1.5)
    f = f[index]
    print(f)
    sxx = sxx[index, :]

    ax2 = plt.subplot(2, 1, 2)
    vals = ax2.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
    ax2.set_xlim([min(csv_times), max(csv_times)])
    ax2.set_xlabel(f"Time (Day Hour:Minute)", fontweight="bold")
    ax2.set_ylabel("Frequency (Hz)", fontweight="bold")
    ax2.axvline(x=arrival_time_rel, c="red")
    cbar = plt.colorbar(vals, orientation="horizontal")
    cbar.set_label("Power ((m/s)^2/sqrt(Hz))", fontweight="bold")

    plt.plot()
    plt.show()

    ...
