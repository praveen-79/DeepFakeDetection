"""
Deepfake Detection Models & Multi-Layer Feature Fusion Package
==============================================================
Implements:
1. Video & Image Preprocessing (OpenCV frame extraction, face cropping, resizing, normalization)
2. Multi-Layer Feature Extraction:
   - MesoNet (Meso-4 CNN for mesoscopic facial manipulation artifacts)
   - XceptionNet (Depthwise Separable Convolutional network for deep texture forensics)
   - Vision Transformer (ViT patch self-attention encoder for global spatial consistency)
3. Feature Fusion (Concatenation of MesoNet, XceptionNet, and ViT feature vectors)
4. Random Forest Classification (Binary classification: Real vs. Fake)
"""

from .feature_extractors import (
    VideoImagePreprocessor,
    MesoNetExtractor,
    XceptionFeatureExtractor,
    VisionTransformerExtractor,
    FeatureFusionClassifier,
    build_mesonet4_keras,
    build_xception_extractor_keras,
    build_vit_extractor_keras,
)

__all__ = [
    "VideoImagePreprocessor",
    "MesoNetExtractor",
    "XceptionFeatureExtractor",
    "VisionTransformerExtractor",
    "FeatureFusionClassifier",
    "build_mesonet4_keras",
    "build_xception_extractor_keras",
    "build_vit_extractor_keras",
]
