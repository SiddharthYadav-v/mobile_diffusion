from md.encoder import Encoder
from md.decoder import Decoder
from md.diffusion import DDPM

from argparse import ArgumentParser

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, ToTensor

from torch.utils.data import DataLoader
from torch import nn
from torch import randn, save, load, randint, tensor
from torch.optim import Adam

from md.clip import CLIP

from tqdm import tqdm

import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_folder",
                        help = "The path to the dataset you plan to use for training the Variational Autoencoder")
    parser.add_argument("--img_size",
                        help = "Final image size after resizing",
                        type = int,
                        default = 256)
    parser.add_argument("--random_flips",
                        help = "Introduce random horizontal flips or not",
                        type = int,
                        default = 1)
    parser.add_argument("--batch_size",
                        help = "Batch size to use for the training data",
                        type = int,
                        default = 16)
    parser.add_argument("--epochs",
                        help = "Number of epochs to train the model for",
                        type = int,
                        default = 20)
    parser.add_argument("--device",
                        help = "Which device to use for training",
                        default = "cpu")
    parser.add_argument("--lr",
                        help = "Learning rate",
                        type = float,
                        default = 1e-5)
    parser.add_argument("--save_as",
                        help = "Filename to save model",
                        default = "ddpm.pt")
    parser.add_argument("--load_decoder_from",
                        help = "File to load VAE decoder",
                        default = "decoder.pt")
    parser.add_argument("--load_encoder_from",
                        help = "File to load VAE encoder",
                        default = "encoder.pt")
    
    args = parser.parse_args()

    data_folder = args.data_folder
    img_size = args.img_size
    random_flips = args.random_flips
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device
    lr = args.lr
    save_as = args.save_as
    load_encoder_from = args.load_encoder_from
    load_decoder_from = args.load_decoder_from

    transforms = [
        ToTensor(),
        Resize((img_size, img_size)),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
    if random_flips:
        transforms.append(RandomHorizontalFlip())
    
    dataset = ImageFolder(args.data_folder, Compose(transforms))
    loader = DataLoader(dataset, shuffle = True, batch_size = batch_size)
    
    encoder = Encoder().to(device)
    try:
        encoder.load_state_dict(load(load_encoder_from))
    except FileNotFoundError:
        print (f"{load_encoder_from} not found, not loading")
    except RuntimeError:
        print (f"{load_encoder_from} Weight and key mismatch, not loading")
    for param in encoder.parameters():
        param.requires_grad_(False)
    
    decoder = Decoder().to(device)
    try:
        decoder.load_state_dict(load(load_decoder_from))
    except FileNotFoundError:
        print (f"{load_decoder_from} not found, not loading")
    except RuntimeError:
        print (f"{load_decoder_from} Weight and key mismatch, not loading")
    for param in decoder.parameters():
        param.requires_grad_(False)
    
    ddpm = DDPM(4, 1000, 512, 128).to(device)
    try:
        ddpm.load_state_dict(load(save_as))
    except FileNotFoundError:
        print (f"{save_as} not found, not loading")
    except RuntimeError:
        print (f"{save_as} Weight and key mismatch, not loading")
        
    clip = CLIP().to(device)
    for param in clip.parameters():
        param.requires_grad_(False)
    
    opt = Adam(ddpm.parameters(), lr = lr)
    loss_fn = nn.MSELoss()
    
    num_params = sum(p.numel() for model in [ddpm, encoder, decoder, clip] for p in model.parameters())
    print ("Number of parameters in model:", num_params)

    for epoch in range(epochs):
        with tqdm(loader, unit = "batch") as iterator:
            for img, _ in iterator:
                iterator.set_description(f"{epoch + 1}")
                img = img.to(device)
                noise = randn(img.shape[0], 4, img_size // 8, img_size // 8).to(device)
                y = randint(0, 76, (img.shape[0], 77,)).to(device)
                t = randint(0, 999, (img.shape[0],)).to(device)

                opt.zero_grad(set_to_none = True)
                emb = clip(y)
                enc = encoder(img, noise)
                noise = randn(img.shape[0], 4, img_size // 8, img_size // 8).to(device)
                eps = ddpm(enc, t, emb, noise)
                dec = decoder(enc)

                loss = loss_fn(eps, noise)

                loss.backward()
                opt.step()
                iterator.set_postfix(loss = loss.item())
                
        save(ddpm.state_dict(), save_as)
