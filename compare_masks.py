#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end mask comparison.

Renders SAM's real segmentation against the adapter's segmentation using the
*same* SAM mask decoder and the *same* point prompt. The only difference is
where the image embedding comes from:

    SAM path:     image -> SAM encoder      -> decoder -> mask   (baseline)
    Adapter path: image -> OWL -> adapter   -> decoder -> mask   (ours)

This is the test that actually tells us whether the adapter is usable, which the
feature-matching loss cannot. Reports per-image IoU (adapter vs SAM) and saves a
side-by-side figure.

Usage:
    python compare_masks.py \
        --sam-checkpoint sam_vit_b.pth \
        --adapter-checkpoint checkpoints/adapter_best.pth \
        --coco-dir data/coco/val2017 \
        --num-samples 5 \
        --output mask_comparison.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from nanosam.adapter import OwlToSamAdapter
from nanosam.encoders import OWLEncoder


def load_adapter(checkpoint_path: str, device: str) -> OwlToSamAdapter:
    """Load the trained adapter weights."""
    adapter = OwlToSamAdapter().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()
    print(f"Loaded adapter from {checkpoint_path}")
    return adapter


@torch.no_grad()
def adapter_image_embedding(image_rgb: np.ndarray, owl: OWLEncoder,
                            adapter: OwlToSamAdapter, device: str) -> torch.Tensor:
    """image (H,W,3 uint8 RGB) -> adapter SAM-style embedding (1,256,64,64)."""
    img = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(device)
    owl_feat = owl(img)          # (1, 576, 768)
    return adapter(owl_feat)     # (1, 256, 64, 64)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boolean masks."""
    a, b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def overlay(ax, image, mask=None, point=None, title=""):
    """Show image with an optional red mask overlay and a green prompt point."""
    ax.imshow(image)
    if mask is not None:
        m = np.zeros((*mask.shape, 4))
        m[mask] = [1.0, 0.0, 0.0, 0.5]  # red, 50% alpha
        ax.imshow(m)
    if point is not None:
        ax.scatter([point[0]], [point[1]], c="lime", s=120,
                   marker="*", edgecolors="black", linewidths=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Adapter vs SAM mask comparison")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--owl-model", default="google/owlvit-base-patch32")
    parser.add_argument("--coco-dir", default="data/coco/val2017",
                        help="Directory of images to sample from")
    parser.add_argument("--image", default=None,
                        help="Single image path (overrides --coco-dir sampling)")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="mask_comparison.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from segment_anything import sam_model_registry, SamPredictor

    device = args.device
    print(f"Device: {device}\n")

    # Load SAM (encoder + decoder) via the predictor — handles all pre/postprocessing
    print(f"Loading SAM {args.sam_model_type}...")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint).to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    # Load OWL encoder + trained adapter
    owl = OWLEncoder(model_name=args.owl_model).to(device)
    adapter = load_adapter(args.adapter_checkpoint, device)

    # Gather images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        all_imgs = sorted(Path(args.coco_dir).glob("*.jpg"))
        if not all_imgs:
            print(f"✗ No images found in {args.coco_dir}")
            sys.exit(1)
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(all_imgs), size=min(args.num_samples, len(all_imgs)), replace=False)
        image_paths = [all_imgs[i] for i in idx]

    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None, :]

    ious = []
    for row, img_path in enumerate(image_paths):
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]
        point = np.array([[w // 2, h // 2]])   # center-point prompt
        labels = np.array([1])

        # --- SAM baseline ---
        predictor.set_image(image)
        sam_masks, sam_scores, _ = predictor.predict(
            point_coords=point, point_labels=labels, multimask_output=False)
        sam_mask = sam_masks[0]

        # --- Adapter path: swap the image embedding, reuse everything else ---
        adapter_feat = adapter_image_embedding(image, owl, adapter, device)
        predictor.features = adapter_feat   # inject adapter embedding
        adp_masks, adp_scores, _ = predictor.predict(
            point_coords=point, point_labels=labels, multimask_output=False)
        adp_mask = adp_masks[0]

        iou = mask_iou(sam_mask, adp_mask)
        ious.append(iou)
        print(f"[{row+1}/{n}] {img_path.name}  IoU(adapter vs SAM) = {iou:.3f}")

        overlay(axes[row][0], image, point=point[0], title=f"Input + prompt\n{img_path.name}")
        overlay(axes[row][1], image, mask=sam_mask, point=point[0], title="SAM (baseline)")
        overlay(axes[row][2], image, mask=adp_mask, point=point[0],
                title=f"Adapter (ours)\nIoU = {iou:.3f}")

    mean_iou = float(np.mean(ious))
    fig.suptitle(f"Adapter vs SAM   |   mean IoU = {mean_iou:.3f}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"\nMean IoU (adapter vs SAM): {mean_iou:.3f}")
    print(f"Saved comparison figure to {args.output}")

    # Plain-language read on the result
    if mean_iou >= 0.75:
        print("→ Strong: adapter masks closely track SAM. Likely usable as-is.")
    elif mean_iou >= 0.5:
        print("→ Moderate: rough object masks, but boundaries are coarse. Try OWL-B/16.")
    else:
        print("→ Weak: masks diverge from SAM. The 24x24 grid is too coarse — needs B/16 or a learned upsampler.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
