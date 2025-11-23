"""
Meat package initialization file.
"""

from .model import MeatIdentifier
from .preprocessing import (
    load_image,
    load_image_cv2,
    preprocess_for_model,
    detect_packaging_features,
    batch_load_images,
    augment_image
)

__version__ = '1.0.0'
__all__ = [
    'MeatIdentifier',
    'load_image',
    'load_image_cv2',
    'preprocess_for_model',
    'detect_packaging_features',
    'batch_load_images',
    'augment_image'
]
