"""
Meat Identifier - Main Inference Script
Load images and classify them using the trained model.
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MeatIdentifier
from preprocessing import load_image, detect_packaging_features


def predict_single_image(model, image_path, show_features=False):
    """
    Predict meat packaging type for a single image.
    
    Args:
        model: Trained MeatIdentifier instance
        image_path: Path to the image file
        show_features: Whether to show packaging feature analysis
        
    Returns:
        Dictionary with prediction results
    """
    print(f"\nAnalyzing: {image_path}")
    
    # Load and preprocess image
    try:
        image = load_image(image_path, target_size=(224, 224))
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Make prediction
    result = model.predict(image)
    
    # Print results
    print(f"Prediction: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nAll probabilities:")
    for class_name, prob in result['all_probabilities'].items():
        print(f"  {class_name}: {prob:.2%}")
    
    # Optional: Show packaging features
    if show_features:
        print("\nPackaging Features Analysis:")
        features = detect_packaging_features(image_path)
        for feature, value in features.items():
            print(f"  {feature}: {value:.4f}")
    
    return result


def predict_batch(model, image_dir, output_file=None):
    """
    Predict meat packaging for all images in a directory.
    
    Args:
        model: Trained MeatIdentifier instance
        image_dir: Directory containing images
        output_file: Optional JSON file to save results
        
    Returns:
        List of prediction results
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_paths = []
    
    # Find all image files
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(f'*{ext}'))
        image_paths.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return []
    
    print(f"Found {len(image_paths)} images")
    
    results = []
    for img_path in image_paths:
        result = predict_single_image(model, str(img_path))
        if result:
            results.append({
                'image': str(img_path),
                'prediction': result
            })
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    class_counts = {}
    for r in results:
        class_name = r['prediction']['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count} images")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Meat Identifier - Classify raw meat packaging in images'
    )
    parser.add_argument(
        'input',
        help='Path to image file or directory containing images'
    )
    parser.add_argument(
        '--model',
        default='models/meat_identifier.h5',
        help='Path to trained model file (default: models/meat_identifier.h5)'
    )
    parser.add_argument(
        '--output',
        help='JSON file to save batch prediction results'
    )
    parser.add_argument(
        '--features',
        action='store_true',
        help='Show packaging feature analysis'
    )
    parser.add_argument(
        '--build-model',
        action='store_true',
        help='Build and save a new untrained model (for testing)'
    )
    
    args = parser.parse_args()
    
    # Initialize model
    identifier = MeatIdentifier()
    
    # Build new model if requested
    if args.build_model:
        print("Building new model...")
        identifier.build_model()
        identifier.compile_model()
        os.makedirs('models', exist_ok=True)
        identifier.save_model(args.model)
        print(f"New model saved to {args.model}")
        print("\nNote: This is an untrained model. You need to train it before use.")
        return
    
    # Load trained model
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("\nTo create a new model structure, run:")
        print(f"  python {sys.argv[0]} --build-model <input_path>")
        print("\nTo train the model, use train.py with your dataset.")
        sys.exit(1)
    
    print(f"Loading model from {args.model}...")
    identifier.load_model(args.model)
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Path not found: {args.input}")
        sys.exit(1)
    
    if input_path.is_file():
        # Single image prediction
        predict_single_image(identifier, str(input_path), show_features=args.features)
    elif input_path.is_dir():
        # Batch prediction
        predict_batch(identifier, str(input_path), output_file=args.output)
    else:
        print(f"Error: Invalid input path: {args.input}")
        sys.exit(1)


if __name__ == '__main__':
    main()
