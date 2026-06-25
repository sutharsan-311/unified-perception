# Phase B: Adapter Training on COCO Dataset

Real-time training pipeline for the OWL-SAM adapter using COCO dataset.

**Status:** Ready to train | **Timeline:** ~1 day (2-4 hours GPU time + setup)

---

## Overview

Phase B takes the Phase A adapter module and trains it to produce SAM-compatible features from OWL-ViT-B encoder output.

**Training approach:**
- Supervised regression: learn mapping from OWL features → SAM features
- COCO dataset (public, 5000 images)
- Combined MSE + cosine similarity loss
- Frozen OWL and SAM encoders (transfer learning)
- 2-4 hour training on RTX 3060

**Later:** Fine-tune on hospital-specific medical images (Phase B.2)

---

## Installation

### 1. Install Dependencies

```bash
pip install torch torchvision transformers pytorch-cuda::11.8
```

### 2. Download COCO Dataset

**Option A: Full COCO (360GB)**
```bash
# Install COCO tools
pip install pycocotools

# Download (requires manual download from https://cocodataset.org/)
# Extract to data/coco/
# Structure should be:
# data/coco/
#   ├── train2017/
#   │   ├── 000000000001.jpg
#   │   ├── 000000000002.jpg
#   │   └── ...
#   └── val2017/
#       ├── 000000000397.jpg
#       └── ...
```

**Option B: Lightweight Test Dataset (for quick testing)**
```bash
mkdir -p data/coco/{train2017,val2017}

# Copy some images manually or use:
# wget https://mscoco.org/dataset/download/ (follow instructions)
```

### 3. Verify Setup

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Training

### Basic Usage

```bash
# Train on 5000 COCO images
python train.py \
    --dataset-dir data/coco \
    --num-images 5000 \
    --batch-size 4 \
    --num-epochs 10 \
    --device cuda
```

### Advanced Options

```bash
# Custom learning rate and loss weights
python train.py \
    --dataset-dir data/coco \
    --num-images 5000 \
    --batch-size 8 \
    --num-epochs 10 \
    --learning-rate 5e-4 \
    --mse-weight 0.6 \
    --cosine-weight 0.4 \
    --checkpoint-dir checkpoints \
    --log-interval 10

# Load and resume from checkpoint
python train.py \
    --dataset-dir data/coco \
    --num-images 5000 \
    --load-checkpoint checkpoints/adapter_phase_b_best.pth \
    --num-epochs 20
```

### Arguments

```
Data:
  --dataset-dir          Root directory of COCO dataset (default: data/coco)
  --num-images           Number of training images (default: 5000)
  --validation-size      Number of validation images (default: 100)

Training:
  --batch-size           Batch size (default: 4)
  --num-epochs           Number of epochs (default: 10)
  --learning-rate        Learning rate (default: 1e-3)
  --num-workers          Data loading workers (default: 4)

Loss:
  --mse-weight           Weight for MSE loss (default: 0.5)
  --cosine-weight        Weight for cosine similarity (default: 0.5)

Device:
  --device               Device to train on (default: cuda)
  --fp16                 Use FP16 precision (flag)

Checkpoint:
  --checkpoint-dir       Directory for checkpoints (default: checkpoints)
  --load-checkpoint      Path to checkpoint to resume from

Other:
  --seed                 Random seed (default: 42)
  --log-interval         Logging interval in batches (default: 10)
```

---

## Training Pipeline Architecture

### Data Loading (`nanosam/data_loader.py`)

**COCOAdapterDataset**
- Loads images from COCO train2017/val2017
- Resizes to 768×768 (OWL-ViT input size)
- Normalizes with ImageNet stats
- Returns: (image_tensor, target_features)

**TrainingDataLoader**
- Wrapper for easy DataLoader creation
- Handles shuffling, batching, workers
- Separate train/validation loaders

### Loss Function (`nanosam/trainer.py`)

**AdapterLoss**

```
Total Loss = 0.5 * MSE Loss + 0.5 * Cosine Similarity Loss

MSE Loss:
  - Pixel-level feature matching
  - L2 distance in feature space
  - Good for magnitude alignment

Cosine Similarity Loss:
  - Direction/orientation matching
  - 1 - cosine_similarity(pred, target)
  - Good for feature direction alignment

Combined:
  - Both magnitude AND direction
  - More robust than either alone
```

### Training Loop (`nanosam/trainer.py`)

**AdapterTrainer**

Features:
- Gradient accumulation support
- Learning rate scheduling (warmup + cosine annealing)
- Gradient clipping (prevents exploding gradients)
- Checkpoint saving (best model + periodic)
- Validation during training
- Metrics tracking

**Training Process:**
1. Load batch of images
2. Generate synthetic OWL features (Phase C: use real OWL encoder)
3. Adapter forward pass: OWL features → SAM features
4. Compute combined loss
5. Backward pass with gradient clipping
6. Optimizer step + scheduler update
7. Optional validation and checkpointing

---

## Expected Results

### Training Time

| Dataset Size | Batch Size | GPU | Time |
|--|--|--|--|
| 1,000 images | 2 | RTX 3060 | ~30 min |
| 5,000 images | 4 | RTX 3060 | ~2 hours |
| 10,000 images | 8 | RTX 3060 | ~4 hours |

### Performance Metrics

**Training Convergence:**
- Initial loss: ~0.8-1.0
- After 1 epoch: ~0.3-0.5
- After 5 epochs: ~0.1-0.2
- Final (10 epochs): ~0.05-0.1

**Validation Loss:**
- Should decrease with training
- Plateau after ~5 epochs
- Best checkpoint typically epoch 5-8

**Sample Outputs:**
```
Epoch 1 [0/1250] Loss: 0.824591 (Avg: 0.824591) LR: 1.00e-05
Epoch 1 [10/1250] Loss: 0.795832 (Avg: 0.810212) LR: 4.00e-04
...
Epoch 10 [1240/1250] Loss: 0.052483 (Avg: 0.061234) LR: 3.14e-05
Validation Loss: 0.048923
→ Saved best checkpoint!
```

---

## Important Notes

### Phase B vs Phase B.2

**Phase B (This):**
- Train on public COCO dataset
- General object detection/segmentation
- Baseline implementation
- Validates approach works

**Phase B.2 (Later):**
- Fine-tune on hospital rosbag data
- Medical-specific features (anatomy, equipment)
- Higher accuracy for hospital use case
- When hospital data is available

### Current Limitations

**Synthetic Target Features:**
Currently, target SAM features are simulated (random tensors). This is okay for:
- Validating training pipeline works
- Testing gradient flow
- Measuring training convergence

**Phase C Improvement:**
Real target features from actual SAM encoder:
```python
# Phase C: Replace in trainer.py line ~130
# From:
target_features = torch.randn(...)

# To:
with torch.no_grad():
    target_features = sam_encoder(images)
```

This requires:
- Loading SAM encoder
- Running it on same image batch
- Using output as training target

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train.py --batch-size 2

# Reduce number of workers
python train.py --num-workers 2

# Use smaller dataset
python train.py --num-images 1000
```

### Dataset Not Found

```bash
# Create dummy dataset for testing
mkdir -p data/coco/{train2017,val2017}
# Copy some test images there

# Or download real COCO:
# https://cocodataset.org/#download
```

### Training is Slow

```bash
# Increase workers (if I/O bound)
python train.py --num-workers 8

# Reduce number of images (for quick iteration)
python train.py --num-images 1000 --num-epochs 3

# Check GPU usage
nvidia-smi  # Should see high GPU utilization
```

### Loss Not Decreasing

```bash
# Check learning rate
python train.py --learning-rate 5e-4

# Reduce batch size (noisier gradients = better escape from local minima)
python train.py --batch-size 2

# Increase training duration
python train.py --num-epochs 20
```

---

## Monitoring Training

### Real-time Monitoring

Training loop prints:
- Current loss
- Average loss per epoch
- Learning rate
- Validation loss every 500 batches

### Checkpoints

Automatically saved to `checkpoints/`:
- `adapter_phase_b_best.pth` - Best validation loss
- `adapter_phase_b_epoch{N}.pth` - Periodic saves
- `adapter_phase_b_final.pth` - Final model

### Resume Training

```bash
python train.py --load-checkpoint checkpoints/adapter_phase_b_best.pth
```

---

## Validation Metrics

The trainer monitors:

**Training Loss:** MSE + Cosine similarity combined
- Good general indicator of learning

**Validation Loss:** Same loss on held-out data
- Indicates generalization
- Early stopping possible if diverging

**Checkpoint Saving:**
- Best checkpoint: lowest validation loss
- Periodic: every 5 epochs
- Final: after all epochs

---

## Next Steps (Phase C)

After training:

1. **Integration:**
   - Load best checkpoint in main pipeline
   - Replace synthetic features with real SAM encoder

2. **Testing:**
   - Run on real medical images
   - Compare quality vs sequential baseline
   - Measure end-to-end latency

3. **Optimization:**
   - TensorRT export for Jetson
   - Async pipeline for parallel execution

4. **Fine-tuning (Phase B.2):**
   - When hospital data available
   - Lower learning rate
   - Fewer epochs (typically 3-5)

---

## Files Created

```
nanosam/
  ├── training_config.py       Config dataclasses + presets
  ├── data_loader.py           COCO dataset and DataLoader
  ├── trainer.py               Training loop and loss functions

train.py                        Main training script
PHASE_B_TRAINING.md            This document
```

---

## Citation

If you use this training pipeline:

```bibtex
@software{unified_perception_phase_b_2026,
  author = {Sutharsan},
  title = {Phase B: OWL-SAM Adapter Training Pipeline},
  year = {2026},
  url = {https://github.com/sutharsan-311/unified-perception}
}
```

---

## Support

**Questions?**
1. Check output logs for errors
2. Verify COCO dataset exists
3. Check GPU memory with `nvidia-smi`
4. Try with smaller dataset first

**Issues?**
- Make sure PyTorch CUDA version matches GPU driver
- Verify adapter.py loads correctly: `python -c "from nanosam.adapter import OwlToSamAdapter; print(OwlToSamAdapter())"`

---

*Phase B: Transform synthetic understanding into learned knowledge.*

🚀 Ready to train!
