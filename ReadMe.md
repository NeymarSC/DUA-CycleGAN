# **DUA-CycleGAN** for Pneumonia Chest Radiographs Image Encryption

## Project Overview

This repository contains the official implementation of **DUA-CycleGAN**, a novel medical image encryption algorithm proposed in our paper: *"Privacy-Secure and Feature-Enhanced Encryption Algorithm for Pneumonia Chest Radiographs"*.

Traditional encryption methods often struggle with the high-resolution nature of medical images, leading to high computational overhead. Meanwhile, standard deep learning-based encryption often fails to preserve critical diagnostic details. To address these issues, we propose a specialized CycleGAN-based framework that integrates an **Adaptive Lesion Feature Enhancement Module** and a **Dynamic Weighted UACI Loss Function**.

## **Key Features**

🧠 **Adaptive Lesion Feature Enhancement:** Combines modified Channel Attention (using Variance-Weighted Pooling) and Spatial Attention to specifically protect sensitive areas like lung fields and lesion boundaries.

**⚙️Dynamic Weighted UACI Loss:** A novel loss function that adaptively adjusts the weight of the Unified Average Changing Intensity (UACI) metric during training. This ensures the ciphertext is highly sensitive to minor changes in the plaintext, effectively resisting differential attacks

📊 **Comprehensive Metrics**: Monitors training progress using PSNR, SSIM, and LPIPS to evaluate reconstruction fidelity.

⚙️ **Flexible Architecture**: Supports multiple generator backbones (ResNet-6/9 blocks, UNet) and discriminator types (PatchGAN).

📝 **Robust Logging**: Automatically saves training logs, loss curves, and visual results for easy debugging and comparison.



## Code Structure

DUA-CycleGAN
│
.gitignore                 # Specifies intentionally untracked files to ignore in Git.`
 LICENSE                   # The open-source software license (e.g., MIT, GPL).`
 README.md                 
   ── checkpoints/              # [Directory] Stores model weights and training logs.`
   └── [experiment_name]/    # Folder named after the specific experiment.
      ├── latest.pth        # Latest saved model weights.`
`│       └── training_logs.txt # Detailed console output and metrics during training.`
`│`
`├── data/                    # [Directory] Dataset loading and processing scripts.`
`│   ├── __init__.py`
`│   ├── aligned_dataset.py   # Loader for paired datasets (Pix2Pix style).`
`│   ├── base_dataset.py      # Abstract base class for custom datasets.`
`│   ├── image_folder.py      # Utility to traverse image directories.`
`│   └── unaligned_dataset.py # Loader for unpaired datasets (CycleGAN default).`
`│`
`├── datasets/                # [Directory] Raw data storage.`

`├── trainA/         # Original chest X-rays`
     `├── trainB/         # Noise domain samples (or dummy images)`
     `├── testA/`
     `└── testB/`

`│   ├── bibtex/              # Example dataset folder.`
`│   └── download_cyclegan_dataset.sh # Script to download standard datasets.`
`│`
`├── docs/                   # [Directory] Extended documentation and guides.`
`│   ├── overview.md          # High-level project architecture explanation.`
`│   └── datasets.md         # Detailed guide on how to prepare datasets.`
`│`
`├── models/                 # [Directory] Core model definitions and logic.`
`│   ├── __init__.py`
`│   ├── base_model.py       # Abstract base class for all models (defines interfaces).`
`│   ├── cycle_gan_model.py  # Implements CycleGAN logic (generators, discriminators, losses).`
`│   └── networks.py         # Defines network architectures (ResNet/UNet, CBAM modules).`
`│`
`├── options/                # [Directory] Command-line argument configurations.`
`│   ├── base_options.py     # Basic arguments shared by training and testing.`
`│   ├── train_options.py    # Arguments specific to the training phase.`
`│   └── test_options.py    # Arguments specific to the testing/inference phase.`
`│`
`├── scripts/                # [Directory] Shell scripts for quick execution.`
`│   ├── train.sh            # One-click script to start training with preset parameters.`
`│   ├── test.sh             # One-click script to run inference on test data.`
`│   └── conda_deps.sh      # Script to install dependencies via Conda.`
 │
 ├── util/                   # [Directory] Helper utilities.`
`│   ├── __init__.py`
`│   ├── html.py             # Generates HTML pages to display training results.`
`│   ├── image_pool.py       # Implements a history buffer for stabilizing GAN training.`
`│   ├── util.py             # General utility functions (e.g., tensor conversions).`
`│   └── visualizer.py       # Handles logging of losses and displaying images (Visdom).`
`│`
`├── train.py                 # Entry point for training the model.`
`├── test.py                  # Entry point for testing/inference.`
`└── environment.yml          # Conda environment configuration file.`

## Quick Start

### 1. Prepare Dataset

Organize your data in the standard CycleGAN unaligned format:

datasets/
└── your_dataset/
    ├── trainA/         # Source domain training images
    ├── trainB/         # Target domain training images
    ├── testA/          # Source domain test images
    └── testB/          # Target domain test images

### 2. Run Training

Example for training a model on the Maps dataset:

python train.py \
  --dataroot ./datasets/pneumonia_xray \
  --name DUA_CycleGAN_pneumonia \
  --model cycle_gan \
  --netG resnet_9blocks \
  --direction AtoB \
  --lambda_A 10.0 \
  --lambda_B 10.0 \
  --lambda_identity 0.5 \
  --batch_size 6 \
  --load_size 286 --crop_size 256

### 3. Common Arguments

| **Argument**        | **Description**                                      | **Default** |
| :------------------ | :--------------------------------------------------- | :---------- |
| `--dataroot`        | Path to the dataset directory                        | Required    |
| `--name`            | Name of the experiment (used for saving logs/models) | Required    |
| `--model`           | Model type (fixed as `cycle_gan`)                    | `cycle_gan` |
| `--direction`       | Translation direction (`AtoB`or `BtoA`)              | `AtoB`      |
| `--lambda_A`        | Weight for cycle consistency loss (A->B->A)          | 10.0        |
| `--lambda_B`        | Weight for cycle consistency loss (B->A->B)          | 10.0        |
| `--lambda_identity` | Weight for identity loss                             | 0.5         |
| `--continue_train`  | Resume training from the latest checkpoint           | False       |

### 4.Evaluation Metrics

During training, the system tracks the following metrics to ensure quality:

| **Metric**  | **Description**                                              | **Goal**                           |
| :---------- | :----------------------------------------------------------- | :--------------------------------- |
| **Entropy** | Information Entropy (Shannon entropy). Measures the randomness and uniformity of pixel distribution in the encrypted image. | Higher is better (close to 8.0)    |
| **NPCR**    | Number of Pixels Change Rate. The percentage of pixels that change when a single pixel in the plaintext is altered. | Higher is better (close to 99.6%)  |
| **UACI**    | Unified Average Changing Intensity. The average intensity difference between two encrypted images generated from slightly different plaintexts. | Higher is better                   |
| **SSIM**    | Structural Similarity Index. Measures the similarity between two images in terms of luminance, contrast, and structure. | Higher is better (closer to 1)     |
| **PSNR**    | Peak Signal-to-Noise Ratio. Measures the quality of reconstructed (decrypted) images compared to the original. | Higher is better (higher dB value) |
| **LPIPS**   | Learned Perceptual Image Patch Similarity. A perceptual metric based on deep learning features; measures perceptual differences. | Lower is better (closer to 0)      |

Logs are saved to `checkpoints/[name]/training_logs.txt`.

### 5.Configuration Guide

#### Network Architectures

Modify the network structure using the following arguments:

| Argument | Options                                                    | Description                        |
| -------- | ---------------------------------------------------------- | ---------------------------------- |
| `--netG` | `resnet_9blocks`, `resnet_6blocks`, `unet_256`, `unet_128` | Generator architecture             |
| `--netD` | `basic`, `n_layers`, `pixel`                               | Discriminator architecture         |
| `--ngf`  | 64, 128, 256                                               | Number of filters in the generator |
| `--norm` | `instance`, `batch`, `none`                                | Normalization layer type           |
