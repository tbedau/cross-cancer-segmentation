"""Create a demo segmentation model with untrained weights.

Generates a UNet model with the correct architecture and number of output
classes so that the inference pipeline can be tested end-to-end without
the real (non-public) model weights.  Predictions will be random noise.

Usage:
    uv run create_demo_model.py
    uv run segment.py --model DEMO --image test_image/example.jpg
"""

from __future__ import annotations

from pathlib import Path

import segmentation_models_pytorch as smp
import torch

# Must match segment.py constants
ENCODER = "timm-efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"
N_CLASSES = 12  # LUNG model (used by Model.DEMO)


def main() -> None:
    """Create a demo model with random weights."""
    output_dir = Path(__file__).resolve().parent / "models" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "demo_model.pth"

    print(f"Creating demo model ({N_CLASSES} classes) ...")
    demo_model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=N_CLASSES,
        in_channels=3,
    )
    torch.save(demo_model, output_path)
    print(f"Saved to {output_path}")
    print("\nRun inference with:\n  uv run segment.py --model DEMO --image <path>")


if __name__ == "__main__":
    main()
