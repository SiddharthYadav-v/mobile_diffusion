from md.clip import CLIP
from md.decoder import Decoder
from md.diffusion import DDPM

from torch import randn, load, no_grad, min as minimum, max as maximum, randint

from cv2 import cvtColor, COLOR_RGB2BGR, imwrite

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("text_prompt",
                        help = "Text prompt to use for sampling")
    parser.add_argument("--num_steps",
                        help = "Number of steps in DDPM sampling",
                        type = int,
                        default = 1000)
    parser.add_argument("--num_images",
                        help = "Number of images to sample",
                        type = int,
                        default = 1)
    parser.add_argument("--device",
                        help = "Which device to use for sampling",
                        default = "cpu")
    parser.add_argument("--load_decoder_from",
                        help = "File to load VAE decoder",
                        default = "decoder.pt")
    parser.add_argument("--load_ddpm_from",
                        help = "File to load DDPM",
                        default = "ddpm.pt")
    
    args = parser.parse_args()
    
    prompt = args.text_prompt
    num_steps = args.num_steps
    num_images = args.num_images
    device = args.device
    load_decoder_from = args.load_decoder_from
    load_ddpm_from = args.load_ddpm_from
    
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
        ddpm.load_state_dict(load(load_ddpm_from))
    except FileNotFoundError:
        print (f"{load_ddpm_from} not found, not loading")
    except RuntimeError:
        print (f"{load_ddpm_from} Weight and key mismatch, not loading")
    for param in ddpm.parameters():
        param.requires_grad_(False)
        
    clip = CLIP().to(device)
    for param in clip.parameters():
        param.requires_grad_(False)
        
    total_params = sum(param.numel() for param in decoder.parameters()) + sum(param.numel() for param in ddpm.parameters()) + sum(param.numel() for param in clip.parameters())
    print ("Total parameters :", total_params)

    with no_grad():
        noise = randn((num_images, 4, 32, 32)).to(device)
        tok = randint(0, 49407, (num_images, 77,)).to(device)
        emb = clip(tok)
        img = ddpm.sample(noise, emb, num_steps, device)
        dec = decoder(img)
        disp = dec[0]
        disp -= minimum(disp)
        disp *= 255 / maximum(disp)
        disp = disp.cpu().numpy().transpose(1, 2, 0).astype("uint8")
        disp = cvtColor(disp, COLOR_RGB2BGR)
        imwrite("ddpm.png", disp)