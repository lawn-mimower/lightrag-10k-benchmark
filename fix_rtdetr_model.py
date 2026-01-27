#!/usr/bin/env python3
"""
Fix RT-DETRv2 model loading issue for Docling OCR
"""

import os
from transformers import AutoModel, AutoProcessor
from huggingface_hub import hf_hub_download

def download_rtdetr_model():
    """Download and cache RT-DETRv2 model to fix loading issues."""

    print("Downloading RT-DETRv2 model components...")

    try:
        # The model used by Docling
        model_name = "PekingU/rtdetr_r50vd"

        # Download model files
        print(f"Downloading {model_name}...")

        # Download config and weights
        config_file = hf_hub_download(repo_id=model_name, filename="config.json")
        model_file = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")

        print(f"Model files downloaded to cache:")
        print(f"  Config: {config_file}")
        print(f"  Model: {model_file}")

        # Try to load the model to verify
        print("\nVerifying model can be loaded...")
        from transformers import RTDetrModel, RTDetrImageProcessor

        processor = RTDetrImageProcessor.from_pretrained(model_name)
        model = RTDetrModel.from_pretrained(model_name)

        print("✓ RT-DETRv2 model loaded successfully!")

    except ImportError:
        print("\n⚠ RT-DETR classes not found in transformers.")
        print("Installing transformers with RT-DETR support...")
        os.system("pip install 'transformers>=4.55.0' --upgrade")
        print("\nPlease run this script again after installation.")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTrying alternative fix...")

        # Alternative: Install from GitHub
        print("Installing latest transformers from source...")
        os.system("pip install git+https://github.com/huggingface/transformers.git")


if __name__ == "__main__":
    download_rtdetr_model()