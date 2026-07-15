# Images Autoencoder (VQ-VAE)

This folder contains the training and decoding scripts for a Vector Quantized Variational Autoencoder (VQ-VAE). It is specifically designed to deeply compress continuous image streams (both standard RGB and Thermal IR).

## Dataset Structure
Before running the training script, ensure your dataset is organized in a folder matching the target mode (`rgb` or `thermal`) in the same directory as the scripts.

```text
.
├── rgb/                # Folder containing RGB training images
├── thermal/            # Folder containing Thermal IR training images
├── vqvae_train_v4.py
└── vqvae_decode_v4.py
```

## 1. Training the Model (`vqvae_train.py`)
This script trains the model, tracks losses, and automatically encodes the entire dataset into compressed `.npy` indices upon completion. 

**Basic Usage:**
```bash
python vqvae_train_v4.py --mode thermal
```

**Arguments:**
* `--mode` : Select the dataset type (`rgb` or `thermal`). **Required.**
* `--epochs` : Set the number of epochs to train (default: `20`).
* `--resume` : Add this flag to resume training from the latest `_best.pth` checkpoint.

**Outputs:**
* `{mode}_vqvae_best.pth` : The saved model weights.
* `{mode}_vqvae_indices.npy` : The highly compressed latent indices.
* `{mode}_vqvae_filenames.txt` : The original filenames mapping for the indices.
* `{mode}_vqvae_training_log.csv` : Log of all tracked loss metrics.
* `{mode}_vqvae_output/` : Folder containing a sample reconstruction grid.

## 2. Decoding the Images (`vqvae_decode.py`)
This script loads the trained model weights and reconstructs the continuous images from the compressed `.npy` indices file.

**Basic Usage (Decode sample of 10 images):**
```bash
python vqvae_decode_v4.py --mode thermal
```

**Decode All Images:**
```bash
python vqvae_decode_v4.py --mode thermal --n 0
```

**Arguments:**
* `--mode` : Select the dataset type (`rgb` or `thermal`). **Required.**
* `--n` : Number of images to decode (default: `10`). Set to `0` to decode the entire dataset.
* `--npy` : Path to custom indices file (default auto-detects from mode).
* `--model` : Path to specific weights file (default auto-detects from mode).
* `--output_dir` : Path to save the reconstructed images (defaults to `{mode}_vqvae_decoded`).
