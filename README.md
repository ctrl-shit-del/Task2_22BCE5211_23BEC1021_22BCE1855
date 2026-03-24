# SGMSE+ Speech De-Reverberation — Task 2

**Samsung Spatial Audio Hackathon**

| Member | Roll Number |
|--------|------------|
| Sailee Allyadwar | 22BCE5211 |
| Navomi S. Ramesh | 22BCE1855 |
| Harsh Singhal | 23BEC1021 |

---

## Overview

This repository contains our implementation for **Task 2** of the Samsung Spatial Audio Hackathon —  
**Blind Speech De-Reverberation** using the **SGMSE+** framework.

SGMSE+ (Score-Based Generative Model for Speech Enhancement) is a diffusion-based model that uses:
- **Backbone**: NCSN++ (Noise Conditional Score Network++)
- **SDE**: OUVE (Ornstein-Uhlenbeck Variance Exploding)
- **Sampler**: Predictor-Corrector (Euler-Maruyama + Langevin) at inference

---

## Repository Structure

```
├── train.py          # Training script (fine-tune or train from scratch)
├── inference.py      # Inference script (de-reverberate .wav files)
├── evaluate.py       # Evaluation script (PESQ, ESTOI, SI-SDR, DNSMOS)
├── prepare_data.py   # Data preparation utilities
├── metrics.csv       # Evaluation results
├── requirements.txt  # Python dependencies
└── Report.pdf        # Project report
```

> **Note:** Model checkpoint files (`.ckpt`) are NOT included due to GitHub's 100 MB file limit.  
> Download them from the link below and place them in the root directory.

---

## 📦 Model Checkpoints (Google Drive)

The trained model weights must be downloaded separately:

| File | Description | Size |
|------|-------------|------|
| `sgmse_task2_light.ckpt` | Lightweight model for inference | ~250 MB |
| `epoch=4-step=10000.ckpt` | Full training checkpoint | ~1.25 GB |

**👉 [Download Checkpoints from Google Drive](YOUR_DRIVE_LINK_HERE)**

> Place both `.ckpt` files in the **root directory** of this repo after downloading.

---

## Setup

### 1. Clone this repo
```bash
git clone <this-repo-url>
cd <repo-name>
```

### 2. Install SGMSE (required dependency)
```bash
git clone https://github.com/sp-uhh/sgmse
cd sgmse && pip install -e .
cd ..
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Training
Train from scratch:
```bash
python train.py \
    --base-dir ./data \
    --epochs 4 \
    --batch-size 4 \
    --lr 1e-4
```

Resume from a checkpoint:
```bash
python train.py \
    --base-dir ./data \
    --resume-from epoch=4-step=10000.ckpt \
    --epochs 8
```

### Inference (De-Reverberation)
Single file:
```bash
python inference.py \
    --ckpt sgmse_task2_light.ckpt \
    --input reverberant.wav \
    --output enhanced.wav
```

Batch (folder):
```bash
python inference.py \
    --ckpt sgmse_task2_light.ckpt \
    --input ./reverberant_wavs/ \
    --output ./enhanced_wavs/ \
    --N 30 --corrector-steps 1 --snr 0.5
```

### Evaluation
```bash
python evaluate.py \
    --clean ./data/test/clean/ \
    --enhanced ./enhanced_wavs/ \
    --output results.csv
```

Metrics computed: **PESQ**, **ESTOI**, **SI-SDR**, **SI-SIR**, **SI-SAR**, **DNSMOS**

---

## Results

See `metrics.csv` for full per-file results. Summary available in `Report.pdf`.

---

## References

- [SGMSE+ Paper — Richter et al., 2022](https://arxiv.org/abs/2208.05830)
- [Official SGMSE Repository](https://github.com/sp-uhh/sgmse)
- [Microsoft DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS)
