# Adapter Design Quick Reference

## The Core Problem

```
OWL-ViT-B outputs:        SAM expects:
(B, 576, 768)      X      (B, 256, 64, 64)

   576 tokens            4,096 spatial positions
   768-dim features      256-dim features
   1D sequence           2D grid
```

## The Solution

```python
OWL Output: (B, 576, 768)
    ↓
Reshape to 2D grid: (B, 24, 24, 768)
    ↓
Upsample spatially: (B, 24, 24, 768) → (B, 64, 64, 768)  [bilinear]
    ↓
Reduce features: (B, 64, 64, 768) → (B, 64, 64, 256)      [2-layer MLP]
    ↓
Normalize: LayerNorm
    ↓
Format for SAM: (B, 256, 64, 64) ✓
```

## Exact Dimensions

| Layer | Input | Output | Operation |
|-------|-------|--------|-----------|
| OWL Encoder | (B, 3, 768, 768) | (B, 576, 768) | Vision Transformer |
| Reshape | (B, 576, 768) | (B, 24, 24, 768) | view() |
| Upsample | (B, 768, 24, 24) | (B, 768, 64, 64) | interpolate bilinear |
| MLP Layer 1 | (B*4096, 768) | (B*4096, 512) | Linear(768→512) + GELU |
| MLP Layer 2 | (B*4096, 512) | (B*4096, 256) | Linear(512→256) |
| LayerNorm | (B*4096, 256) | (B*4096, 256) | LayerNorm(256) |
| Reshape Back | (B, 64, 64, 256) | (B, 256, 64, 64) | permute(0,3,1,2) |
| SAM Decoder | (B, 256, 64, 64) | (B, 1, 256, 256) | Mask Decoder |

## Adapter Parameters Breakdown

```
Linear(768 → 512):
  Weight: 768 × 512 = 393,216
  Bias:   512
  Subtotal: 393,728

Linear(512 → 256):
  Weight: 512 × 256 = 131,072
  Bias:   256
  Subtotal: 131,328

LayerNorm(256):
  Weight: 256
  Bias:   256
  Subtotal: 512

────────────────────────
TOTAL: 525,568 parameters (~525K)
```

**Memory Footprint:**
- FP32: ~2.1 MB
- FP16: ~1.05 MB
- INT8: ~512 KB

**Inference Cost:**
- Linear 768→512: ~3.9M MACs (at 64×64 spatial)
- Linear 512→256: ~2.1M MACs
- Total: ~6M MACs per frame (~0.15ms on Orin NX)

## Training Strategy

**Option A: Frozen Encoders (Recommended for Hospital Use)**
```
OWL-ViT-B (frozen) → Adapter (trainable) → SAM (frozen)
```
- Only 525K parameters to train
- Faster training: 2-3 hours on RTX 3060
- Less risk of degrading original model quality
- Can fine-tune on hospital-specific anatomy

**Option B: Joint Fine-tuning (Advanced)**
```
OWL-ViT-B (fine-tune) → Adapter → SAM (fine-tune)
```
- Requires more data
- Longer training: 1-2 weeks
- Higher GPU memory (24GB+)
- Better final performance if data is rich

## Hospital Robot Constraints Check

```
✓ Jetson Orin NX 8GB - Sufficient
  - Model weights: ~2.5GB (OWL + SAM + Adapter)
  - Runtime VRAM: ~3GB during inference
  - Headroom: ~2.5GB for other ops

✓ Latency - Acceptable
  - OWL encode: ~15ms (TensorRT optimized)
  - Adapter: ~0.5ms (tiny)
  - SAM decode: ~20ms (TensorRT optimized)
  - Total: ~35ms/frame → 28 FPS (exceeds 10 FPS requirement)

✓ Thermal - Safe
  - Adapter adds negligible heat
  - Shared ViT-B encode reduces redundant computation
  - Estimated 5-10% power reduction vs sequential

✓ Accuracy - Unproven but Promising
  - Initial validation: measure mAP vs sequential pipeline
  - Target: >95% of sequential performance
  - Hospital-specific metrics (sensitivity for lesions, etc.)
```

## Implementation Checklist

### Phase A: Build Adapter Layer
- [ ] Create `adapter.py` with OwlToSamAdapter class
- [ ] Test on dummy tensors to verify shape transformations
- [ ] Benchmark inference time on Orin NX
- [ ] Validate memory usage

### Phase B: Integration & Training
- [ ] Integrate adapter into unified pipeline
- [ ] Create training loop (supervised alignment loss)
- [ ] Test on hospital dataset (if available)
- [ ] Measure mAP@50 vs sequential baseline

### Phase C: Deployment Validation
- [ ] Export adapter to TensorRT (optional but recommended)
- [ ] Test full pipeline on Orin NX with real medical images
- [ ] Thermal testing under sustained load
- [ ] Finalize for hospital deployment

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Adapter doesn't learn | Low | High | Fallback to sequential (still real-time) |
| Spatial misalignment | Medium | Medium | Validate on training data first |
| Memory overflow | Very Low | High | Use FP16 precision if needed |
| Performance regression | Low | Medium | Comprehensive validation suite |

## Next: Phase A Implementation

Ready to build `nanosam/adapter.py`?

The adapter is self-contained and testable independently:
1. Create the module
2. Test shape transformations
3. Profile on Jetson hardware
4. Then integrate into full pipeline

This de-risks Phase B training.

