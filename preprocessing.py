"""
Image preprocessing utilities for meat identification.
Handles loading, preprocessing, and augmenting images.
"""

import cv2
import numpy as np
from PIL import Image
import os


def load_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image from file.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
        
    Returns:
        Preprocessed image array (normalized to 0-1 range)
    """
    try:
        # Load image using PIL
        img = Image.open(image_path)
        
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to target size
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        return img_array
    
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")


def load_image_cv2(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image using OpenCV.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
        
    Returns:
        Preprocessed image array (normalized to 0-1 range)
    """
    try:
        # Load image using OpenCV
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize to 0-1 range
        img_array = img.astype(np.float32) / 255.0
        
        return img_array
    
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")


def preprocess_for_model(image_array):
    """
    Additional preprocessing specific to the model requirements.
    
    Args:
        image_array: Numpy array of the image
        
    Returns:
        Preprocessed image ready for model input
    """
    # Ensure the array is in the correct shape
    if len(image_array.shape) == 2:
        # Grayscale to RGB
        image_array = np.stack([image_array] * 3, axis=-1)
    
    # Ensure values are in 0-1 range
    if image_array.max() > 1.0:
        image_array = image_array / 255.0
    
    return image_array


def detect_packaging_features(image_path):
    """
    Detect visual features that indicate packaging type.
    This can help improve classification accuracy.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with detected features
    """
    img = cv2.imread(image_path)
    if img is None:
        return {}
    
    features = {}
    
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check for reflective surfaces (aluminum foil)
    # Aluminum typically has high brightness variance
    brightness = hsv[:, :, 2]
    features['brightness_std'] = float(np.std(brightness))
    features['brightness_mean'] = float(np.mean(brightness))
    
    # Check for ice crystals or frost (frozen meat)
    # Often appears as white spots with high local variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features['edge_variance'] = float(laplacian_var)
    
    # Check for transparency/plastic (vacuum packaging)
    # Vacuum packages often have wrinkles and consistent texture
    features['saturation_mean'] = float(np.mean(hsv[:, :, 1]))
    
    # Color analysis - raw meat typically has red/brown hues
    red_channel = img[:, :, 2]
    blue_channel = img[:, :, 0]
    features['red_blue_ratio'] = float(np.mean(red_channel) / (np.mean(blue_channel) + 1))
    
    return features


def batch_load_images(image_paths, target_size=(224, 224)):
    """
    Load multiple images at once.
    
    Args:
        image_paths: List of image file paths
        target_size: Target size for resizing
        
    Returns:
        Numpy array of preprocessed images
    """
    images = []
    valid_paths = []
    
    for path in image_paths:
        try:
            img = load_image(path, target_size)
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Warning: Could not load {path}: {str(e)}")
            continue
    
    if not images:
        raise ValueError("No valid images could be loaded")
    
    return np.array(images), valid_paths


def augment_image(image_array):
    """
    Apply random augmentation to an image.
    Useful for testing model robustness.
    
    Args:
        image_array: Input image array (0-1 range)
        
    Returns:
        Augmented image array
    """
    img = (image_array * 255).astype(np.uint8)
    
    # Random rotation
    if np.random.random() > 0.5:
        angle = np.random.randint(-30, 30)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
    
    # Random brightness adjustment
    if np.random.random() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        img = np.clip(img * factor, 0, 255).astype(np.uint8)
    
    # Random flip
    if np.random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    return img.astype(np.float32) / 255.0


def save_preprocessed_image(image_array, output_path):
    """
    Save a preprocessed image to disk.
    
    Args:
        image_array: Image array (0-1 range)
        output_path: Path to save the image
    """
    # Convert to 0-255 range
    img = (image_array * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Save image
    cv2.imwrite(output_path, img_bgr)
    print(f"Image saved to {output_path}")
