import numpy as np

import matplotlib.pyplot as plt

import a_generate_spectrogram as a


def spectrogram(t, f, sxx):
    plt.pcolormesh(t, f, sxx)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.show()


def main():
    shown_moon = False

    for file in a.recursive_files(a.PREPROCESS_DIR, ext_filter=".npz"):
        if shown_moon:
            if "mars" not in file:
                continue

        d = np.load(file)
        spec_f, spec_t, sxx = d["spec_f"], d["spec_t"], d["sxx"]

        print(file)
        spectrogram(spec_t, spec_f, sxx)

        shown_moon = True


if __name__ == "__main__":
    main()
