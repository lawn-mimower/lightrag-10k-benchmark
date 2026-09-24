#!/bin/bash

echo "Updating transformers to support RT-DETRv2..."

# Update transformers to latest version (4.55+)
pip install --upgrade "transformers>=4.55.0"

# Also update related packages
pip install --upgrade "tokenizers>=0.21"
pip install --upgrade accelerate
pip install --upgrade onnxruntime

echo "Done! Transformers updated to support RT-DETRv2"