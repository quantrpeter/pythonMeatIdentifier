"""
Training script for the meat identification model.
Trains the model on a custom dataset organized in folders.
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MeatIdentifier


def train_model(train_dir, val_dir, epochs=20, batch_size=32, learning_rate=0.0001):
    """
    Train the meat identification model.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        
    Expected directory structure:
        train_dir/
            no_meat/
                image1.jpg
                image2.jpg
                ...
            vacuum_packaged/
                image1.jpg
                image2.jpg
                ...
            aluminum_foil/
                image1.jpg
                image2.jpg
                ...
            frozen/
                image1.jpg
                image2.jpg
                ...
    """
    print("="*60)
    print("Meat Identifier Training")
    print("="*60)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize model
    print("\nInitializing model...")
    identifier = MeatIdentifier()
    
    # Build model architecture
    print("Building model architecture...")
    identifier.build_model(trainable_layers=20)
    
    # Compile model
    print("Compiling model...")
    identifier.compile_model(learning_rate=learning_rate)
    
    # Print model summary
    print("\nModel Summary:")
    identifier.summary()
    
    # Train the model
    print(f"\nStarting training...")
    print(f"Training directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    history = identifier.train(
        train_data_dir=train_dir,
        validation_data_dir=val_dir,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Save the final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/meat_identifier_{timestamp}.h5'
    identifier.save_model(model_path)
    
    # Also save as default model
    identifier.save_model('models/meat_identifier.h5')
    
    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    }
    
    history_path = f'models/training_history_{timestamp}.json'
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")
    
    # Print final metrics
    print("\n" + "="*60)
    print("Final Training Metrics")
    print("="*60)
    print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
    
    return identifier, history


def main():
    parser = argparse.ArgumentParser(
        description='Train the Meat Identifier model on a custom dataset'
    )
    parser.add_argument(
        '--train-dir',
        required=True,
        help='Path to training data directory (should contain class subdirectories)'
    )
    parser.add_argument(
        '--val-dir',
        required=True,
        help='Path to validation data directory (should contain class subdirectories)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0001,
        help='Learning rate for optimizer (default: 0.0001)'
    )
    
    args = parser.parse_args()
    
    # Validate directories exist
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory not found: {args.train_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.val_dir):
        print(f"Error: Validation directory not found: {args.val_dir}")
        sys.exit(1)
    
    # Train the model
    train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == '__main__':
    main()
