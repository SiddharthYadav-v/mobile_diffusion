from md.encoder import Encoder
from md.decoder import Decoder

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, ToTensor

from torch import randn, load, no_grad, randint, cat, min as minimum, max as maximum

from cv2 import imwrite, cvtColor, COLOR_RGB2BGR

from os.path import exists

if __name__ == "__main__":
    device = "cuda:2"
    transforms = [
        ToTensor(),
        Resize((256, 256)),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        RandomHorizontalFlip()
    ]

    dataset = ImageFolder("data/CelebAMask-HQ/CelebA-HQ-img", Compose(transforms))
    encoder = Encoder().to(device)
    if exists("encoder.pt"):
        encoder.load_state_dict(load("encoder.pt", map_location = device))
    decoder = Decoder().to(device)
    if exists("decoder.pt"):
        decoder.load_state_dict(load("decoder.pt", map_location = device))
        
    with no_grad():
        img, _ = dataset[randint(0, len(dataset) - 1, (1,))]
        noise = randn(1, 4, 32, 32).to(device)
        enc = encoder(img.unsqueeze(0).to(device), noise)
        dec = decoder(enc)
        save_img = cat([img.to(device), dec[0]], dim = -1)
        save_img -= minimum(save_img)
        save_img *= 255 / maximum(save_img)
        save_img = save_img.cpu().numpy().transpose(1, 2, 0).astype("uint8")
        save_img = cvtColor(save_img, COLOR_RGB2BGR)
        imwrite("reconstruction.png", save_img)
