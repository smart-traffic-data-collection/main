import os
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance


# ==========================================
# MODEL (must match vqvae_train.py exactly)
# ==========================================
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings  = num_embeddings
        self.embedding_dim   = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = dist.argmin(1)
        z_q = self.embedding(indices).view(B, H, W, C).permute(0, 3, 1, 2)
        loss_vq     = F.mse_loss(z_q.detach(), z)
        loss_commit = F.mse_loss(z_q, z.detach()) * self.commitment_cost
        z_q_st = z + (z_q - z).detach()
        return z_q_st, loss_vq + loss_commit, indices.view(B, H, W)


class VQVAEAutoencoder(nn.Module):
    def __init__(self, in_channels=3, num_embeddings=512, embedding_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, embedding_dim, 1),
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def decode_from_indices(self, indices):
        B, H, W = indices.shape
        z_q = self.vq.embedding(indices.view(-1)).view(B, H, W, self.vq.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2)
        return self.decoder(z_q)

    def forward(self, x):
        z = self.encoder(x)
        z_q, loss, indices = self.vq(z)
        return self.decoder(z_q), loss, indices

# ==========================================
# POST-PROCESSING (RGB only)
# ==========================================
def enhance_decoded(tensor):
    img = transforms.ToPILImage()(tensor.cpu())
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Color(img).enhance(1.15)
    return img

# ==========================================
# DECODE
# ==========================================
def decode(args):
    is_thermal = (args.mode == 'thermal')
    in_channels = 1 if is_thermal else 3
    prefix = args.mode

    npy_path = args.npy or f'{prefix}_vqvae_indices.npy'
    filenames_path = args.filenames or f'{prefix}_vqvae_filenames.txt'
    model_path = args.model or f'{prefix}_vqvae_best.pth'
    output_dir = args.output_dir or f'{prefix}_vqvae_decoded'
    n = args.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode        : {args.mode.upper()}")
    print(f"Device      : {device}")

    model = VQVAEAutoencoder(in_channels=in_channels).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(
        ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    )
    model.eval()
    print(f"Loaded model: {model_path}")

    indices_all = np.load(npy_path).astype(np.int64)  # (N, 32, 32)
    with open(filenames_path) as f:
        filenames = f.read().splitlines()

    print(f"Indices     : {indices_all.shape} dtype=uint16")
    print(f"Storage     : {indices_all.astype(np.uint16).nbytes/1024:.1f} KB for {len(filenames)} images")
    print(f"vs raw      : {512*512*in_channels*len(filenames)/1024:.0f} KB uncompressed\n")

    if args.stride > 1:
        indices_all = indices_all[::args.stride]
        filenames = filenames[::args.stride]
        print(f"Stride      : {args.stride} -> {len(filenames)} images after striding\n")

    if n > 0:
        indices_all = indices_all[:n]
        filenames = filenames[:n]

    os.makedirs(output_dir, exist_ok=True)
    total = len(filenames)
    batch_size = 32
    all_recons = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = torch.tensor(indices_all[start:end], dtype=torch.long, device=device)
        with torch.no_grad():
            recon = model.decode_from_indices(batch).cpu().float()
        all_recons.append(recon)

    all_recons = torch.cat(all_recons, dim=0)

    for i, fname in enumerate(filenames):
        recon = all_recons[i]
        if is_thermal:
            vutils.save_image(recon, os.path.join(output_dir, f'decoded_{fname}'))
        else:
            enhance_decoded(recon).save(
                os.path.join(output_dir, f'decoded_{fname}'), quality=95)
        print(f"  [{i+1:04d}/{total}] {fname}")

    # Overview grid (first 10)
    grid = make_grid(all_recons[:min(10, total)], nrow=5, padding=4, pad_value=1.0)
    vutils.save_image(grid, os.path.join(output_dir, f'grid_top{min(10,total)}.png'))
    print(f"\n{total} images -> '{output_dir}/'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decode VQ-VAE indices to images')
    parser.add_argument('--mode', type=str, required=True, choices=['rgb', 'thermal'])
    parser.add_argument('--npy', type=str, default=None)
    parser.add_argument('--filenames', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--n', type=int, default=10,
                        help='Images to decode (0=all, default=10)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Take every Nth image from the dataset (default=1, no stride)')
    args = parser.parse_args()
    decode(args)
