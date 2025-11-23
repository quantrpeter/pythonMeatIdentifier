"""
Example usage script demonstrating the Meat Identifier API.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meat import MeatIdentifier, load_image


def example_build_and_save_model():
    """Example: Build and save a new model."""
    print("=" * 60)
    print("Example 1: Building and Saving a Model")
    print("=" * 60)
    
    # Initialize the identifier
    identifier = MeatIdentifier()
    
    # Build the model architecture
    identifier.build_model(trainable_layers=20)
    
    # Compile the model
    identifier.compile_model(learning_rate=0.0001)
    
    # Print model summary
    identifier.summary()
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    identifier.save_model('models/example_model.h5')
    
    print("\n✓ Model built and saved successfully!")


def example_load_and_predict():
    """Example: Load a model and make predictions."""
    print("\n" + "=" * 60)
    print("Example 2: Loading Model and Making Predictions")
    print("=" * 60)
    
    # Check if model exists
    model_path = 'models/meat_identifier.h5'
    if not os.path.exists(model_path):
        print(f"\nModel not found at {model_path}")
        print("Please train a model first or build one using example 1")
        return
    
    # Initialize and load model
    identifier = MeatIdentifier()
    identifier.load_model(model_path)
    
    # Example: Load and predict an image
    # (Replace with actual image path)
    image_path = 'test_images/sample.jpg'
    
    if os.path.exists(image_path):
        # Load image
        image = load_image(image_path, target_size=(224, 224))
        
        # Make prediction
        result = identifier.predict(image)
        
        # Print results
        print(f"\nPrediction: {result['class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nAll class probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.2%}")
    else:
        print(f"\nTest image not found at {image_path}")
        print("To test predictions, add an image to test_images/sample.jpg")


def example_training_workflow():
    """Example: Complete training workflow."""
    print("\n" + "=" * 60)
    print("Example 3: Training Workflow (Conceptual)")
    print("=" * 60)
    
    print("\nTo train the model, organize your data like this:")
    print("""
    dataset/
    ├── train/
    │   ├── no_meat/
    │   ├── vacuum_packaged/
    │   ├── aluminum_foil/
    │   └── frozen/
    └── validation/
        ├── no_meat/
        ├── vacuum_packaged/
        ├── aluminum_foil/
        └── frozen/
    """)
    
    print("\nThen run:")
    print("python train.py --train-dir dataset/train --val-dir dataset/validation --epochs 20")
    
    print("\nThis will:")
    print("1. Load and augment your training images")
    print("2. Train the model with early stopping")
    print("3. Save the best model based on validation accuracy")
    print("4. Save training history for analysis")


def example_batch_prediction():
    """Example: Batch prediction on multiple images."""
    print("\n" + "=" * 60)
    print("Example 4: Batch Prediction")
    print("=" * 60)
    
    print("\nTo process multiple images:")
    print("python predict.py path/to/image_folder/")
    
    print("\nTo save results to JSON:")
    print("python predict.py path/to/image_folder/ --output results.json")
    
    print("\nThe output will include:")
    print("- Individual predictions for each image")
    print("- Summary statistics by class")
    print("- Optional JSON export of all results")


def example_custom_prediction():
    """Example: Custom prediction with preprocessing."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Prediction with Preprocessing")
    print("=" * 60)
    
    from meat.preprocessing import detect_packaging_features
    
    image_path = 'test_images/sample.jpg'
    
    if os.path.exists(image_path):
        # Analyze packaging features
        features = detect_packaging_features(image_path)
        
        print("\nPackaging Feature Analysis:")
        print(f"  Brightness STD: {features.get('brightness_std', 'N/A')}")
        print(f"  Brightness Mean: {features.get('brightness_mean', 'N/A')}")
        print(f"  Edge Variance: {features.get('edge_variance', 'N/A')}")
        print(f"  Saturation Mean: {features.get('saturation_mean', 'N/A')}")
        print(f"  Red/Blue Ratio: {features.get('red_blue_ratio', 'N/A')}")
        
        print("\nThese features help identify:")
        print("  - High brightness variance → Aluminum foil (reflective)")
        print("  - High edge variance → Frozen (ice crystals)")
        print("  - Red/Blue ratio > 1 → Raw meat (red/brown color)")
    else:
        print(f"\nTest image not found at {image_path}")
        print("Add an image to test this feature")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("MEAT IDENTIFIER - USAGE EXAMPLES")
    print("=" * 60)
    
    # Run examples
    example_build_and_save_model()
    example_load_and_predict()
    example_training_workflow()
    example_batch_prediction()
    example_custom_prediction()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nFor more information, see README.md")


if __name__ == '__main__':
    main()
