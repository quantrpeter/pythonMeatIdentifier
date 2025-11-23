#!/usr/bin/env python3
"""
Setup script to organize images into proper directory structure for training.
Creates the required class subdirectories and helps organize images.
"""

import os
import shutil
from pathlib import Path


def setup_directory_structure(base_dir='data'):
    """
    Create the required directory structure for training.
    
    Creates:
        data/
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
    """
    
    class_names = ['no_meat', 'vacuum_packaged', 'aluminum_foil', 'frozen']
    splits = ['train', 'validation']
    
    print("Creating directory structure...")
    print("=" * 60)
    
    for split in splits:
        split_path = Path(base_dir) / split
        split_path.mkdir(parents=True, exist_ok=True)
        
        for class_name in class_names:
            class_path = split_path / class_name
            class_path.mkdir(exist_ok=True)
            print(f"✓ Created: {class_path}")
    
    print("=" * 60)
    print("Directory structure created successfully!")
    return class_names


def move_loose_images(base_dir='data'):
    """
    Move any images that are directly in train/validation folders
    into a temporary 'uncategorized' folder for manual sorting.
    """
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.avif', '.webp'}
    moved_files = []
    
    for split in ['train', 'validation']:
        split_path = Path(base_dir) / split
        
        if not split_path.exists():
            continue
        
        # Find images directly in the split directory
        for file_path in split_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # Create uncategorized folder
                uncategorized_path = split_path / 'uncategorized'
                uncategorized_path.mkdir(exist_ok=True)
                
                # Move the file
                new_path = uncategorized_path / file_path.name
                shutil.move(str(file_path), str(new_path))
                moved_files.append((file_path.name, split, 'uncategorized'))
                print(f"Moved: {file_path.name} → {split}/uncategorized/")
    
    return moved_files


def print_instructions():
    """Print instructions for organizing images."""
    
    print("\n" + "=" * 60)
    print("NEXT STEPS: Organize Your Images")
    print("=" * 60)
    
    print("""
You now have the correct directory structure. To train the model:

1. **Organize your images into class folders:**

   Move images from 'uncategorized' folders into the appropriate class:
   
   - no_meat/        → Images WITHOUT any meat
   - vacuum_packaged/  → Meat in vacuum-sealed transparent packages
   - aluminum_foil/    → Meat wrapped in aluminum foil
   - frozen/           → Frozen meat (with ice, frost, or in freezer bags)

2. **Minimum images per class:**
   - Training: At least 20-50 images per class (more is better!)
   - Validation: At least 5-10 images per class

3. **Example organization:**
   
   Move your current images:
   - data/train/uncategorized/*.jpg → data/train/vacuum_packaged/
   - data/validation/uncategorized/*.jpg → data/validation/vacuum_packaged/

4. **Once organized, run training:**
   
   python train.py --train-dir data/train --val-dir data/validation --epochs 20

5. **Need more images?**
   - Search online for "vacuum sealed meat", "frozen meat", etc.
   - Take photos yourself
   - Use data augmentation (handled automatically during training)
""")
    
    print("=" * 60)


def check_current_state(base_dir='data'):
    """Check and report the current state of the data directories."""
    
    print("\n" + "=" * 60)
    print("CURRENT DATA DIRECTORY STATUS")
    print("=" * 60)
    
    class_names = ['no_meat', 'vacuum_packaged', 'aluminum_foil', 'frozen', 'uncategorized']
    
    for split in ['train', 'validation']:
        split_path = Path(base_dir) / split
        
        if not split_path.exists():
            print(f"\n{split.upper()}: Directory not found")
            continue
        
        print(f"\n{split.upper()}:")
        
        for class_name in class_names:
            class_path = split_path / class_name
            
            if not class_path.exists():
                continue
            
            # Count images
            image_count = len(list(class_path.glob('*.*')))
            
            if image_count > 0:
                status = "✓" if image_count >= 10 else "⚠"
                print(f"  {status} {class_name}: {image_count} images")
            else:
                print(f"  ✗ {class_name}: 0 images (empty)")
    
    print("\n" + "=" * 60)


def main():
    """Main setup function."""
    
    print("\n" + "=" * 60)
    print("MEAT IDENTIFIER - DATA SETUP TOOL")
    print("=" * 60)
    
    base_dir = 'data'
    
    # Create directory structure
    setup_directory_structure(base_dir)
    
    print("\n")
    
    # Move loose images to uncategorized
    moved_files = move_loose_images(base_dir)
    
    if moved_files:
        print(f"\n✓ Moved {len(moved_files)} images to 'uncategorized' folders")
    else:
        print("\n✓ No loose images found")
    
    # Check current state
    check_current_state(base_dir)
    
    # Print instructions
    print_instructions()
    
    print("\nSetup complete! Follow the instructions above to organize your images.")


if __name__ == '__main__':
    main()
