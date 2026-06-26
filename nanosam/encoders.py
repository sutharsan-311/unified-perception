import torch
import torch.nn as nn
import torch.nn.functional as F


class OWLEncoder(nn.Module):
    """Frozen OWL-ViT-B encoder — returns patch features (B, 576, 768)."""

    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    IMAGE_SIZE = 768

    def __init__(self, model_name: str = "google/owlvit-base-patch32"):
        super().__init__()
        from transformers import OwlViTForObjectDetection
        print(f"Loading OWL encoder: {model_name}")
        model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.vision_model = model.owlvit.vision_model
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self.register_buffer('mean', torch.tensor(self.CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self.CLIP_STD).view(1, 3, 1, 1))
        print("  OWL encoder loaded and frozen.")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) in [0, 1] range
        Returns:
            (B, 576, 768) patch features
        """
        if images.shape[-2:] != (self.IMAGE_SIZE, self.IMAGE_SIZE):
            images = F.interpolate(images, (self.IMAGE_SIZE, self.IMAGE_SIZE), mode='bilinear', align_corners=False)
        images = (images - self.mean) / self.std
        outputs = self.vision_model(pixel_values=images)
        # Apply the vision transformer's post layer-norm, then drop the CLS token
        hidden = self.vision_model.post_layernorm(outputs.last_hidden_state)
        return hidden[:, 1:, :]  # (B, 576, 768)


class SAMEncoder(nn.Module):
    """Frozen SAM image encoder — returns spatial features (B, 256, 64, 64)."""

    SAM_MEAN = [123.675, 116.28, 103.53]
    SAM_STD = [58.395, 57.12, 57.375]
    IMAGE_SIZE = 1024

    def __init__(self, checkpoint_path: str, model_type: str = "vit_b"):
        super().__init__()
        from segment_anything import sam_model_registry
        print(f"Loading SAM encoder: {model_type} from {checkpoint_path}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.image_encoder = sam.image_encoder
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self.register_buffer('mean', torch.tensor(self.SAM_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self.SAM_STD).view(1, 3, 1, 1))
        print("  SAM encoder loaded and frozen.")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) in [0, 1] range
        Returns:
            (B, 256, 64, 64) image features
        """
        images = images * 255.0
        if images.shape[-2:] != (self.IMAGE_SIZE, self.IMAGE_SIZE):
            images = F.interpolate(images, (self.IMAGE_SIZE, self.IMAGE_SIZE), mode='bilinear', align_corners=False)
        images = (images - self.mean) / self.std
        return self.image_encoder(images)
