from md.encoder import Encoder
from md.decoder import Decoder

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, ToTensor

from torch import randn, load, no_grad, randint, cat, min as minimum, max as maximum

from cv2 import imwrite, cvtColor, COLOR_RGB2BGR

from os.path import exists

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_folder",
                        help = "Folder to load data for encoding")
    parser.add_argument("--device",
                        help = "Which device to use for sampling",
                        default = "cpu")
    parser.add_argument("--load_decoder_from",
                        help = "File to load VAE decoder",
                        default = "decoder.pt")
    parser.add_argument("--load_encoder_from",
                        help = "File to load VAE encoder",
                        default = "encoder.pt")
    
    args = parser.parse_args()
    
    data_folder = args.data_folder
    device = args.device
    load_decoder_from = args.load_decoder_from
    load_encoder_from = args.load_encoder_from
    
    transforms = [
        ToTensor(),
        Resize((256, 256)),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        RandomHorizontalFlip()
    ]

    dataset = ImageFolder(data_folder, Compose(transforms))
    encoder = Encoder().to(device)
    try:
        encoder.load_state_dict(load(load_encoder_from))
    except FileNotFoundError:
        print (f"{load_encoder_from} not found, not loading")
    except RuntimeError:
        print (f"{load_encoder_from} Weight and key mismatch, not loading")
    decoder = Decoder().to(device)
    try:
        decoder.load_state_dict(load(load_decoder_from))
    except FileNotFoundError:
        print (f"{load_decoder_from} not found, not loading")
    except RuntimeError:
        print (f"{load_decoder_from} Weight and key mismatch, not loading")
        
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
