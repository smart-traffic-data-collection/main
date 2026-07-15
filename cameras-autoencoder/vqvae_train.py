import os
import itertools
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import csv
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image

# ==========================================
# 1. VECTOR QUANTIZATION LAYER
# ==========================================
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        # z: (B, C, H, W) → flatten to (B*H*W, C)
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (BHW, C)

        # Distances to codebook
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = dist.argmin(1)

        # Quantize
        z_q = self.embedding(indices).view(B, H, W, C).permute(0, 3, 1, 2)  # (B,C,H,W)

        # Losses
        loss_vq     = F.mse_loss(z_q.detach(), z)
        loss_commit = F.mse_loss(z_q, z.detach()) * self.commitment_cost

        # Straight-through gradient
        z_q_st = z + (z_q - z).detach()

        indices_2d = indices.view(B, H, W)
        return z_q_st, loss_vq + loss_commit, indices_2d


# ==========================================
# 2. VQ-VAE MODEL
# ==========================================
class VQVAEAutoencoder(nn.Module):
    def __init__(self, in_channels=3, num_embeddings=512, embedding_dim=256):
        super().__init__()

        # --- ENCODER ---
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32,  4, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32,  64,  4, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64,  128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, embedding_dim, 1),
        )

        # --- VECTOR QUANTIZER ---
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)

        # --- DECODER ---
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(),
            nn.ConvTranspose2d(32,  in_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode_to_indices(self, x):
        z = self.encoder(x)
        _, _, indices = self.vq(z)
        return indices

    def decode_from_indices(self, indices):
        B, H, W = indices.shape
        z_q = self.vq.embedding(indices.view(-1)).view(B, H, W, self.vq.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2)
        return self.decoder(z_q)

    def forward(self, x):
        z = self.encoder(x)
        z_q, loss, indices = self.vq(z)
        recon = self.decoder(z_q)
        return recon, loss, indices


# ==========================================
# 3. PERCEPTUAL LOSS (VGG)
# ==========================================
class PerceptualLoss(nn.Module):
    def __init__(self, grayscale=False):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.grayscale = grayscale

    def forward(self, pred, target):
        if self.grayscale:
            pred   = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        return F.l1_loss(self.features(self.normalize(pred)),
                         self.features(self.normalize(target)))


# ==========================================
# 4. DATASET
# ==========================================
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir   = img_dir
        self.img_names = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.img_names[idx])
        img  = Image.open(path).convert('L' if self.transform and hasattr(self.transform, '_is_thermal') else 'RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.img_names[idx]


# ==========================================
# 5. TRAINING
# ==========================================
def train(mode='rgb', resume=False, epochs=20):
    is_thermal  = (mode == 'thermal')
    in_channels = 1 if is_thermal else 3
    input_dir   = mode
    output_dir  = f'{mode}_vqvae_output'
    ckpt_path   = f'{mode}_vqvae_best.pth'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: {mode.upper()} | Device: {device}")

    # Transforms
    if is_thermal:
        train_tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
        val_tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    full_ds = ImageDataset(input_dir, transform=train_tf)
    n_val   = max(1, int(0.1 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    # Re-apply val transform to val split
    class ValWrapper(Dataset):
        def __init__(self, subset, transform):
            self.subset    = subset
            self.transform = transform
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            img_path = os.path.join(input_dir, self.subset.dataset.img_names[self.subset.indices[idx]])
            img = Image.open(img_path).convert('L' if is_thermal else 'RGB')
            return self.transform(img), self.subset.dataset.img_names[self.subset.indices[idx]]

    val_ds = ValWrapper(val_ds, val_tf)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model      = VQVAEAutoencoder(in_channels=in_channels, num_embeddings=512, embedding_dim=256).to(device)
    perceptual = PerceptualLoss(grayscale=is_thermal).to(device)
    optimizer  = optim.Adam(
        itertools.chain(model.encoder.parameters(), model.vq.parameters(), model.decoder.parameters()),
        lr=2e-4
    )
    scaler = GradScaler()

    patience      = 7
    best_val_loss = float('inf')
    no_improve    = 0
    log_rows      = []

    # resume logic
    start_epoch = 0
    if resume:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch   = ckpt['epoch'] + 1
                best_val_loss = ckpt['best_val_loss']
                no_improve    = 0
                print(f"Resuming from epoch {start_epoch} | best val loss: {best_val_loss:.4f}")
            else:
                model.load_state_dict(ckpt)
                print("Loaded weights-only checkpoint. Using reduced lr=5e-5.")
                for g in optimizer.param_groups:
                    g['lr'] = 5e-5
        else:
            print(f"[WARN] --resume set but '{ckpt_path}' not found. Starting fresh.")

    print(f"Train: {n_train} | Val: {n_val} | Epochs: {epochs}\n")
    print(f"Compression: 512×512×{in_channels} → 32×32 indices (uint16)")
    orig_bytes = 512 * 512 * in_channels
    comp_bytes = 32 * 32 * 2
    print(f"Ratio: {orig_bytes/comp_bytes:.0f}× | {orig_bytes} B → {comp_bytes} B per image\n")

    for epoch in range(start_epoch, start_epoch + epochs):
        # --- TRAIN ---
        model.train()
        t_recon = t_vq = t_perc = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                recon, vq_loss, _ = model(imgs)
                l_recon = F.l1_loss(recon, imgs)
                l_perc  = perceptual(recon.float(), imgs.float()) * 0.1
                loss    = l_recon + vq_loss + l_perc
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_recon += l_recon.item()
            t_vq    += vq_loss.item()
            t_perc  += l_perc.item()

        n_t = len(train_loader)
        t_recon /= n_t; t_vq /= n_t; t_perc /= n_t

        # --- VAL ---
        model.eval()
        v_recon = v_vq = v_perc = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                with autocast('cuda'):
                    recon, vq_loss, _ = model(imgs)
                    l_recon = F.l1_loss(recon, imgs)
                    l_perc  = perceptual(recon.float(), imgs.float()) * 0.1
                v_recon += l_recon.item()
                v_vq    += vq_loss.item()
                v_perc  += l_perc.item()

        n_v = len(val_loader)
        v_recon /= n_v; v_vq /= n_v; v_perc /= n_v
        v_total = v_recon + v_vq + v_perc

        print(f"Epoch [{epoch+1:02d}/{start_epoch+epochs}] "
              f"Train L1:{t_recon:.4f} VQ:{t_vq:.4f} Perc:{t_perc:.4f} | "
              f"Val L1:{v_recon:.4f} VQ:{v_vq:.4f} Perc:{v_perc:.4f}")

        log_rows.append({
            'epoch':       epoch+1,
            'train_l1':   round(t_recon, 5), 'train_vq': round(t_vq, 5), 'train_perc': round(t_perc, 5),
            'val_l1':     round(v_recon, 5), 'val_vq':   round(v_vq, 5), 'val_perc':   round(v_perc, 5),
        })

        # Early stopping + save
        if v_total < best_val_loss:
            best_val_loss = v_total
            no_improve    = 0

            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss':        best_val_loss,
            }, ckpt_path)
            print(f"Best model saved (val loss: {best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # --- SAVE CSV LOG ---
    csv_path = f'{mode}_vqvae_training_log.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader(); writer.writerows(log_rows)
    print(f"\nTraining log saved → {csv_path}")

    print(f"\nEncoding all images to indices...")
    _ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(_ckpt['model_state_dict'] if isinstance(_ckpt, dict) and 'model_state_dict' in _ckpt else _ckpt)
    model.eval()

    full_val_loader = DataLoader(
        ImageDataset(input_dir, transform=val_tf),
        batch_size=16, shuffle=False, num_workers=4
    )

    all_indices   = []
    all_filenames = []
    with torch.no_grad():
        for imgs, names in full_val_loader:
            imgs = imgs.to(device)
            idx  = model.encode_to_indices(imgs)
            all_indices.append(idx.cpu().numpy().astype(np.uint16))
            all_filenames.extend(names)

    indices_arr = np.concatenate(all_indices, axis=0)
    np.save(f'{mode}_vqvae_indices.npy', indices_arr)

    with open(f'{mode}_vqvae_filenames.txt', 'w') as f:
        f.write('\n'.join(all_filenames))

    print(f"Saved indices: {indices_arr.shape} dtype={indices_arr.dtype}")
    print(f"{mode}_vqvae_indices.npy ({indices_arr.nbytes/1024:.1f} KB for {len(all_filenames)} images)")
    print(f"{mode}_vqvae_filenames.txt")

    # --- SIDE-BY-SIDE RECONSTRUCTION GRID ---
    print(f"\nGenerating reconstruction grid...")
    sample_imgs, sample_names = next(iter(DataLoader(
        ImageDataset(input_dir, transform=val_tf),
        batch_size=8, shuffle=False
    )))
    sample_imgs = sample_imgs.to(device)
    with torch.no_grad():
        recons, _, _ = model(sample_imgs)

    pairs = torch.cat([sample_imgs.cpu(), recons.cpu()], dim=0)
    grid  = make_grid(pairs, nrow=8, padding=4, pad_value=1.0)
    save_image(grid, os.path.join(output_dir, 'reconstruction_grid.png'))
    print(f"Grid saved → {output_dir}/reconstruction_grid.png")
    print(f"\nDone. Weights → {ckpt_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',   type=str, default='rgb', choices=['rgb', 'thermal'])
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from saved checkpoint')
    parser.add_argument('--epochs', type=int, default=20,
                    help='Number of (additional) epochs to train')
    args = parser.parse_args()
    train(mode=args.mode, resume=args.resume, epochs=args.epochs)
