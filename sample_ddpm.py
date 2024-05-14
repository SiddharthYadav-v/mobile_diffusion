from md.encoder import Encoder
from md.decoder import Decoder
from md.diffusion import DDPM

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, ToTensor

from torch import randn, load, no_grad, randint

import matplotlib.pyplot as plt

from os.path import exists

if __name__ == "__main__":
    device = "cuda:2"
    transforms = [
        ToTensor(),
        Resize((256, 256)),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        RandomHorizontalFlip()
    ]

    dataset = ImageFolder("data/celeba", Compose(transforms))
    encoder = Encoder().to(device)
    if exists("encoder.pt"):
        encoder.load_state_dict(load("encoder.pt", map_location = device))
    decoder = Decoder().to(device)
    if exists("decoder.pt"):
        decoder.load_state_dict(load("decoder.pt", map_location = device))
    ddpm = DDPM(4, 1000, 512, 128).to(device)
    # if exists("ddpm.pt"):
    #     ddpm.load_state_dict(load("ddpm.pt", map_location = device))

    with no_grad():
        noise = randn((1, 4, 32, 32)).to(device) * 0.18215
        img = ddpm.sample(noise, device)
        dec = decoder(img)
        plt.imshow(dec[0].cpu().numpy().transpose(1, 2, 0))
        plt.savefig("ddpm.png")
        plt.close()