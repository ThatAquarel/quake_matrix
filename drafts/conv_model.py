import torch

from torch import nn


class QuakeCRNN(nn.Module): ...


def main(): ...


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")

    main()
