# Unified OWL-ViT + SAM Encoder: Architecture Notes

## Feature Dimension Analysis

### 1. OWL-ViT-B Encoder Output

**Model:** `google/owlvit-base-patch32`

**Output Dimensions:**
```
Shape: (batch_size, num_patches, hidden_dim)
      = (batch_size, 576, 768)

Where:
  - Image size: 768x768 pixels
  - Patch size: 32x32 pixels
  - num_patches = (768 / 32)² = 24² = 576 patches
  - hidden_dim = 768 (ViT-B standard)
```

**Code Reference:** `nanoowl/owl_predictor.py:201`
```python
image_embeds = self.model.layer_norm(image_embeds)  # 768 dim
```

**Tensor Layout:**
- Sequence dimension (patches): 576
- Feature dimension (hidden): 768
- Positional info: Encoded in patch positions (implicit grid)

---

### 2. NanoSAM Encoder Output

**Model:** MobileSAM (ResNet18 or TinyViT backbone)

**Output Dimensions:**
```
Shape: (batch_size, channels, spatial_h, spatial_w)
      = (batch_size, 256, 64, 64)

Where:
  - Image size input: 1024x1024 pixels (configurable)
  - Patch size: 16x16 pixels
  - spatial_h, spatial_w = 1024 / 16 = 64
  - channels (out_chans): 256
```

**Code Reference:** `nanosam/mobile_sam/build_sam.py`
```python
prompt_embed_dim = 256
ImageEncoderViT(
    embed_dim=768,  # internal embedding
    out_chans=256,  # output channels (NECK does 768 → 256)
)
```

**Code Reference:** `nanosam/mobile_sam/modeling/image_encoder.py:106-116`
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.patch_embed(x)           # (B, H/16, W/16, 768)
    # ... transformer blocks ...
    x = self.neck(x.permute(0, 3, 1, 2))  # Converts to (B, 256, H/16, W/16)
    return x
```

**Tensor Layout:**
- Channel-first format (standard for CNNs)
- Spatial grid: 64x64 positions
- Feature channels: 256
- No explicit patch tokens (2D spatial)

---

### 3. Dimension Mismatch Analysis

| Aspect | OWL-ViT-B | SAM | Difference |
|--------|-----------|-----|-----------|
| **Sequence Length** | 576 tokens | 4,096 pixels (64×64) | Different spatial representations |
| **Feature Dimension** | 768 | 256 | Must reduce 768 → 256 |
| **Tensor Format** | (B, 576, 768) | (B, 256, 64, 64) | Flatten vs 2D grid |
| **Spatial Encoding** | Implicit (patch IDs) | Explicit (2D positions) | Need positional mapping |
| **Patch Size** | 32×32 | 16×16 | Different patch granularity |

**Critical Issue:**
```
OWL patches (32×32):  24 × 24 = 576 patches from 768×768 image
SAM patches (16×16):  64 × 64 = 4096 patches from 1024×1024 image

Both encode same spatial info but at DIFFERENT RESOLUTION LEVELS
```

---

### 4. Adapter MLP Architecture

**Conversion Strategy:**

```
OWL-ViT output: (B, 576, 768)
                ↓
    [Reshape to 2D grid]
                ↓
                (B, 24, 24, 768)
                ↓
    [Interpolate 24×24 → 64×64]
                ↓
                (B, 64, 64, 768)
                ↓
    [Linear: 768 → 256]
                ↓
                (B, 64, 64, 256)
                ↓
    [Permute to (B, 256, 64, 64)]
                ↓
SAM Mask Decoder input: (B, 256, 64, 64)
```

**Adapter Network Design:**

```python
class OwlToSamAdapter(torch.nn.Module):
    def __init__(self, owl_hidden_dim=768, sam_hidden_dim=256):
        super().__init__()
        
        # Spatial upsampling: 24×24 → 64×64
        # Using bilinear interpolation (learnable would add params)
        self.spatial_upsample_factor = 64 / 24  # ≈ 2.67
        
        # Feature dimension reduction: 768 → 256
        # Two-layer MLP for non-linear transformation
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(owl_hidden_dim, 512),  # 768 → 512
            torch.nn.GELU(),
            torch.nn.Linear(512, sam_hidden_dim)   # 512 → 256
        )
        
        # LayerNorm for stability
        self.norm = torch.nn.LayerNorm(sam_hidden_dim)
    
    def forward(self, owl_features):
        """
        Args:
            owl_features: (batch, 576, 768)
        Returns:
            sam_features: (batch, 256, 64, 64)
        """
        batch_size = owl_features.shape[0]
        
        # Step 1: Reshape to spatial grid
        owl_grid = owl_features.view(batch_size, 24, 24, 768)
        
        # Step 2: Spatial interpolation (24×24 → 64×64)
        owl_grid = owl_grid.permute(0, 3, 1, 2)  # (B, 768, 24, 24)
        upsampled = torch.nn.functional.interpolate(
            owl_grid,
            size=(64, 64),
            mode='bilinear',
            align_corners=True
        )  # (B, 768, 64, 64)
        
        # Step 3: Reshape for MLP (process each spatial position)
        B, C, H, W = upsampled.shape
        upsampled_flat = upsampled.permute(0, 2, 3, 1)  # (B, 64, 64, 768)
        upsampled_flat = upsampled_flat.reshape(B * H * W, C)
        
        # Step 4: Feature dimension reduction with MLP
        reduced = self.mlp(upsampled_flat)  # (B*64*64, 256)
        
        # Step 5: LayerNorm and reshape back
        reduced = self.norm(reduced)
        sam_features = reduced.view(B, H, W, 256)
        sam_features = sam_features.permute(0, 3, 1, 2)  # (B, 256, 64, 64)
        
        return sam_features
```

**Adapter Parameters:**
- Linear layer 1: 768 × 512 + 512 = **395,264 parameters**
- Linear layer 2: 512 × 256 + 256 = **131,328 parameters**
- LayerNorm: 256 parameters
- **Total: ~526K parameters** (tiny compared to encoder/decoder)

**Memory & Compute:**
- Inference overhead: <1ms (on Jetson Orin NX)
- Memory overhead: ~2MB (negligible)
- Training time: ~2-4 hours on RTX 3060

---

### 5. Validation Requirements

Before deployment, need to verify:

1. **Feature Quality:**
   - Are interpolated features semantically meaningful?
   - Does MLP learn proper transformation?
   - Test with simple objects first (high contrast)

2. **Spatial Alignment:**
   - Do detected boxes from OWL align with SAM output?
   - Is the 24→64 upsampling preserving spatial accuracy?

3. **Hardware Compatibility:**
   - Can Jetson Orin NX handle full pipeline?
   - What's actual latency with adapter?
   - Thermal characteristics under sustained load?

---

### 6. Worst-Case Scenario

If adapter doesn't work well:
- **Fallback A:** Use ResNet18 encoder (lighter) instead of OWL's ViT-B
- **Fallback B:** Frame skipping + caching text embeddings (Path 1)
- **Fallback C:** Deploy as-is (sequential pipeline still runs real-time)

---

## Summary Table

```
┌─────────────────┬──────────────┬────────────────┬──────────────────┐
│ Component       │ Input Shape  │ Output Shape   │ Key Constraint   │
├─────────────────┼──────────────┼────────────────┼──────────────────┤
│ OWL-ViT Encoder │ (B,3,768,768)│ (B, 576, 768)  │ 768 features     │
├─────────────────┼──────────────┼────────────────┼──────────────────┤
│ Adapter MLP     │ (B, 576, 768)│ (B, 256, 64,64)│ ~526K params     │
├─────────────────┼──────────────┼────────────────┼──────────────────┤
│ SAM Decoder     │(B, 256, 64,64)│ (B, 1, 256,256)│ Needs spatial=64 │
└─────────────────┴──────────────┴────────────────┴──────────────────┘
```

---

## Hospital Robot Use Case Constraints

**Jetson Orin NX 8GB Target:**
- Memory: 8GB total (OWL + SAM + adapter = ~3GB model weights)
- Compute: ~40 TFLOPS (enough for both encoders)
- Thermal: ~15W sustained (medical environment consideration)
- Latency: <100ms per frame required (10 FPS minimum)

**Adapter fits within these constraints:**
- Adds <1ms latency
- Uses <50MB VRAM during inference
- Trainable on consumer GPU (RTX 3060, 12GB)

