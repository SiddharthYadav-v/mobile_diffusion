from md.encoder import Encoder
from md.decoder import Decoder
from md.diffusion import DDPM

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, ToTensor

from torch import randn, load, no_grad, min as minimum, max as maximum

from os.path import exists

from cv2 import cvtColor, COLOR_RGB2BGR, imwrite, VideoWriter

if __name__ == "__main__":
    device = "cuda:3"
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
    if exists("ddpm.pt"):
        ddpm.load_state_dict(load("ddpm.pt", map_location = device))
        
    total_params = sum(param.numel() for param in encoder.parameters()) + sum(param.numel() for param in decoder.parameters()) + sum(param.numel() for param in ddpm.parameters())
    print ("Total parameters :", total_params)

    with no_grad():
        noise = randn((1, 4, 32, 32)).to(device) * 0.85
        img = ddpm.sample(noise, device)
        dec = decoder(img)
        disp = dec[0]
        disp -= minimum(disp)
        disp *= 255 / maximum(disp)
        disp = disp.cpu().numpy().transpose(1, 2, 0).astype("uint8")
        disp = cvtColor(disp, COLOR_RGB2BGR)
        imwrite("ddpm.png", disp)