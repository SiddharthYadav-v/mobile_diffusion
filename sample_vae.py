from sd.encoder import VAE_Encoder
from sd.decoder import VAE_Decoder
from argparse import ArgumentParser

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, ToTensor

from torch.utils.data import DataLoader
from torch import randn, save, load, no_grad
from torch.optim import Adam

from tqdm import tqdm

import matplotlib.pyplot as plt

from os.path import exists

if __name__ == "__main__":
    device = "cuda:1"
    transforms = [
        ToTensor(),
        Resize((128, 128)),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        RandomHorizontalFlip()
    ]

    dataset = ImageFolder("data/train", Compose(transforms))
    encoder = VAE_Encoder().to(device)
    if exists("encoder.pt"):
        encoder.load_state_dict(load("encoder.pt"))
    decoder = VAE_Decoder().to(device)
    if exists("decoder.pt"):
        decoder.load_state_dict(load("decoder.pt"))

    with no_grad():
        img, _ = dataset[0]
        _, ax = plt.subplots(1, 2, figsize = (8, 16))
        noise = randn((1, 4, 32, 32)).to(device)
        enc = encoder(img.unsqueeze(0).to(device), noise)
        dec = decoder(enc)
        ax[0].imshow((img.cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0)
        ax[1].imshow((dec[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0)
        plt.savefig("reconstruction.png")
        plt.close()