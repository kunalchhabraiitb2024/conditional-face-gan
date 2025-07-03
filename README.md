# Conditional Face Generation from Embeddings

This project implements a production-ready conditional GAN that generates human faces from embeddings. It demonstrates best practices in model design, evaluation, and experiment tracking, and is suitable for both rapid prototyping and production scaling.

## Problem Statement

- Train a generative model that takes face embeddings as input and outputs 128x128 face images.
- Demonstrate zero-shot generalization on unseen embeddings.
- Train from scratch within 6 hours on available hardware.

## Solution Overview

- **Encoder**: Pre-trained FaceNet (frozen) for robust embeddings.
- **Generator**: Conditional generator that takes embeddings + noise.
- **Discriminator**: Evaluates both image quality and embedding consistency.
- **Losses**: Adversarial + strong reconstruction (MSE) loss.
- **Evaluation**: FID, embedding consistency, diversity, and more (see `metrics.py`).
- **Experiment Tracking**: Full wandb integration and automated reporting (`create_wandb_report.py`).

## Key Features

- **Zero-shot generalization**: Model tested on unseen embeddings.
- **Production metrics**: FID, embedding consistency, diversity, and more.
- **Reproducibility**: All code, configs, and checkpoints included.
- **Clean repo**: Only essential files, with all intermediate artifacts and caches removed.
- **Automated reporting**: Script to generate public wandb reports for submission.

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

### 3. Run evaluation and inference
```bash
python inference_evaluation.ipynb  # Or open in Jupyter/VSCode
```

### 4. Generate a public wandb report
```bash
python create_wandb_report.py
```

## Project Structure
```
model.py            # GAN architectures (Encoder, Generator, Discriminator)
dataset.py          # Custom face dataset and dataloader
train.py            # Training loop and checkpointing
metrics.py          # Advanced GAN evaluation metrics
inference_evaluation.ipynb  # End-to-end inference, metrics, and demo notebook
create_wandb_report.py      # Automated wandb report generation
requirements.txt    # All dependencies
results/            # Model checkpoints and sample outputs
face_dataset/       # Input face images
```

## Evaluation Metrics
- **Fréchet Inception Distance (FID)**: Measures quality/diversity of generated images (lower is better)
- **Embedding Consistency Score**: Cosine similarity between real and generated embeddings (higher is better)
- **Diversity Score**: Measures variety in generated samples (higher is better)
- **MSE Reconstruction Loss**: Pixel-level similarity (lower is better)
- **Zero-shot Generalization**: Performance on unseen embeddings

## Example Results
- FID (10 epochs): ~85
- Embedding Consistency: 0.65 ± 0.15
- Zero-shot Consistency: 0.62

## Compute Used
- **Hardware**: MacBook Pro M1 (CPU only)
- **Training Time**: 30 minutes (10 epochs)
- **Dataset**: CelebA subset (5k samples)

## Reproducibility & Production
- All code and configs included
- Only essential files in repo (no __pycache__, temp, or debug logs)
- Automated evaluation and reporting
- Ready for scaling to larger datasets and longer training

## Wandb Dashboard
- [View Public Dashboard](https://wandb.ai/kunalchhabraiitb-alloan-ai/face-generation-verification/runs/dgjwmtrh)

## License
MIT

---

For details on architecture, experiments, and results, see the notebook and code comments. For production deployment, see the recommendations in the notebook and this README.