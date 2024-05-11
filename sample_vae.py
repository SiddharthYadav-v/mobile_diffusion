from md.encoder import Encoder
from md.decoder import Decoder

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, ToTensor

from torch import randn, load, no_grad, randint

import matplotlib.pyplot as plt

from os.path import exists

if __name__ == "__main__":
    device = "cuda:1"
    transforms = [
        ToTensor(),
        Resize((256, 256)),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        RandomHorizontalFlip()
    ]

    dataset = ImageFolder("data/celeba", Compose(transforms))
    encoder = Encoder().to(device)
    if exists("encoder.pt"):
        encoder.load_state_dict(load("encoder.pt"))
    decoder = Decoder().to(device)
    if exists("decoder.pt"):
        decoder.load_state_dict(load("decoder.pt"))

    with no_grad():
        img, _ = dataset[randint(0, len(dataset) - 1, (1,))]
        _, ax = plt.subplots(1, 2, figsize = (8, 16))
        noise = randn((1, 4, 32, 32)).to(device)
        enc = encoder(img.unsqueeze(0).to(device), noise)
        dec = decoder(enc)
        ax[0].imshow((img.cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0)
        ax[1].imshow((dec[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0)
        plt.savefig("reconstruction.png")
        plt.close()