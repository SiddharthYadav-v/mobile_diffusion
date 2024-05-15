from md.encoder import Encoder
from md.decoder import Decoder

from argparse import ArgumentParser

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, ToTensor

from torch.utils.data import DataLoader
from torch import nn
from torch import randn, save, load, no_grad, cat, min as minimum, max as maximum
from torch.optim import Adam

from tqdm import tqdm

from cv2 import imwrite, cvtColor, COLOR_RGB2BGR

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
    parser.add_argument("--save_encoder_as",
                        help = "Filename to save encoder",
                        default = "encoder.pt")
    parser.add_argument("--save_decoder_as",
                        help = "Filename to save decoder",
                        default = "decoder.pt")
    
    args = parser.parse_args()

    data_folder = args.data_folder
    img_size = args.img_size
    random_flips = args.random_flips
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device
    lr = args.lr
    save_encoder_as = args.save_encoder_as
    save_decoder_as = args.save_decoder_as

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
        encoder.load_state_dict(load(save_encoder_as))
    except FileNotFoundError:
        print (f"{save_encoder_as} not found, not loading")
    except RuntimeError:
        print (f"{save_encoder_as} Weight and key mismatch, not loading")
    
    decoder = Decoder().to(device)
    try:
        decoder.load_state_dict(load(save_decoder_as))
    except FileNotFoundError:
        print (f"{save_decoder_as} not found, not loading")
    except RuntimeError:
        print (f"{save_decoder_as} Weight and key mismatch, not loading")
    
    opt = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        with tqdm(loader, unit = "batch") as iterator:
            for img, _ in iterator:
                iterator.set_description(f"{epoch + 1}")
                img = img.to(device)
                noise = randn(img.shape[0], 4, img_size // 8, img_size // 8).to(device)

                opt.zero_grad(set_to_none = True)
                enc = encoder(img, noise)
                dec = decoder(enc)

                kl = encoder.kl
                mse = loss_fn(dec, img)

                loss = 0.00001 * kl + mse
                loss.backward()
                opt.step()
                iterator.set_postfix(loss = loss.item(), kl = kl.item(), mse = mse.item())
                
        save(encoder.state_dict(), save_encoder_as)
        save(decoder.state_dict(), save_decoder_as)
        
        with no_grad():
            img, _ = next(iter(loader))
            noise = randn(img.shape[0], 4, img.shape[2] // 8, img.shape[3] // 8).to(device)
            enc = encoder(img.to(device), noise)
            dec = decoder(enc)
            save_img = cat([img[0].to(device), dec[0]], dim = -1)
            save_img -= minimum(save_img)
            save_img *= 255 / maximum(save_img)
            save_img = save_img.cpu().numpy().transpose(1, 2, 0).astype("uint8")
            save_img = cvtColor(save_img, COLOR_RGB2BGR)
            imwrite("reconstruction.png", save_img)
