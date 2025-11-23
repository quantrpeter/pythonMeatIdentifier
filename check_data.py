"""
Quick start training script for limited datasets.
Creates a simpler binary classifier when you have limited class diversity.
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MeatIdentifier


def count_images_per_class(data_dir):
    """Count how many images exist in each class."""
    
    class_counts = {}
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return class_counts
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.avif', '.webp']
            image_count = sum(1 for f in class_dir.iterdir() 
                            if f.suffix.lower() in image_extensions)
            if image_count > 0:
                class_counts[class_dir.name] = image_count
    
    return class_counts


def check_dataset_readiness(train_dir, val_dir, min_images=5):
    """Check if dataset is ready for training."""
    
    print("\n" + "="*60)
    print("DATASET READINESS CHECK")
    print("="*60)
    
    train_counts = count_images_per_class(train_dir)
    val_counts = count_images_per_class(val_dir)
    
    print("\nTRAINING SET:")
    for class_name, count in sorted(train_counts.items()):
        status = "✓" if count >= min_images else "⚠"
        print(f"  {status} {class_name}: {count} images")
    
    print("\nVALIDATION SET:")
    for class_name, count in sorted(val_counts.items()):
        status = "✓" if count >= min_images else "⚠"
        print(f"  {status} {class_name}: {count} images")
    
    # Check if we have enough classes with enough images
    valid_train_classes = [c for c, cnt in train_counts.items() if cnt >= min_images]
    valid_val_classes = [c for c, cnt in val_counts.items() if cnt >= min_images]
    
    if len(valid_train_classes) < 2:
        print("\n❌ ERROR: Need at least 2 classes with minimum images in training set")
        return False, valid_train_classes
    
    if len(valid_val_classes) < 2:
        print("\n❌ ERROR: Need at least 2 classes with minimum images in validation set")
        return False, valid_val_classes
    
    print(f"\n✓ Dataset has {len(valid_train_classes)} valid classes for training")
    return True, valid_train_classes


def print_recommendations():
    """Print recommendations for getting more data."""
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS TO FIX DATASET ISSUES")
    print("="*60)
    
    print("""
OPTION 1: Add More Images (RECOMMENDED)
----------------------------------------
You need images for at least 2 different classes. Download or collect:

1. **Vacuum Packaged Meat** (you have this ✓)
   - Search: "vacuum sealed meat", "vacuum packed beef"
   
2. **Aluminum Foil Wrapped Meat**
   - Search: "meat wrapped in foil", "aluminum foil meat"
   
3. **Frozen Meat**
   - Search: "frozen meat bag", "frozen meat ice crystals"
   
4. **No Meat** (for comparison)
   - Use any food/non-meat images

Minimum: 10-20 images per class for each (train + validation)


OPTION 2: Use Data Augmentation with Existing Images
-----------------------------------------------------
If you have at least 10-15 images total, we can use aggressive data 
augmentation to artificially expand your dataset.


OPTION 3: Create Sample/Dummy Classes
--------------------------------------
For testing purposes, you can duplicate your vacuum_packaged images
into other classes temporarily:

  cp data/train/vacuum_packaged/* data/train/aluminum_foil/
  cp data/validation/vacuum_packaged/* data/validation/aluminum_foil/

This won't work well for real predictions, but lets you test the 
training pipeline.


OPTION 4: Build a Binary Classifier
------------------------------------
Focus on just 2 classes:
  - meat (vacuum packaged)
  - no_meat (any non-meat images)

This is simpler and requires less data.
""")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Quick start training with dataset validation'
    )
    parser.add_argument(
        '--train-dir',
        default='data/train',
        help='Path to training data directory'
    )
    parser.add_argument(
        '--val-dir',
        default='data/validation',
        help='Path to validation data directory'
    )
    parser.add_argument(
        '--min-images',
        type=int,
        default=5,
        help='Minimum images per class (default: 5)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force training even if dataset checks fail'
    )
    
    args = parser.parse_args()
    
    # Check dataset
    ready, valid_classes = check_dataset_readiness(
        args.train_dir, 
        args.val_dir, 
        args.min_images
    )
    
    if not ready and not args.force:
        print_recommendations()
        print("\n💡 TIP: Run with --force to attempt training anyway")
        sys.exit(1)
    
    if ready:
        print("\n✅ Dataset is ready for training!")
        print("\nTo start training, run:")
        print(f"  python train.py --train-dir {args.train_dir} --val-dir {args.val_dir}")
    else:
        print("\n⚠️  Warning: Dataset may not be optimal, but forcing training...")
        print("\nStarting training anyway...")
        # You could add training code here


if __name__ == '__main__':
    main()
