"""
Image downloader for meat identification training dataset.
Downloads images from Google using SerpAPI for each category.
"""

import os
import requests
import time
from pathlib import Path
from urllib.parse import urlencode
import json

# Web https://serpapi.com/dashboard
# Your SerpAPI key
SERPAPI_KEY = 'YOURKEY'

# Search queries for each category
SEARCH_QUERIES = {
    'vacuum_packaged': [
        'vacuum sealed meat',
        'vacuum packed beef',
        'vacuum sealed chicken',
        'vacuum packaged meat',
        'cryovac meat',
    ],
    'aluminum_foil': [
        'meat wrapped in aluminum foil',
        'foil wrapped meat',
        'meat in foil',
        'aluminum foil packaged meat',
        'butcher foil meat',
    ],
    'frozen': [
        'frozen meat package',
        'frozen meat bag',
        'frozen meat ice crystals',
        'frozen chicken package',
        'frozen beef package',
    ],
    'no_meat': [
        'vegetables',
        'fruits in package',
        'packaged food no meat',
        'dairy products',
        'bread and bakery',
    ]
}


def search_images(query, num_images=10):
    """
    Search for images using SerpAPI (Google Images).
    
    Args:
        query: Search query string
        num_images: Number of images to retrieve (max 100)
    
    Returns:
        List of image URLs
    """
    
    params = {
        'engine': 'google_images',
        'q': query,
        'api_key': SERPAPI_KEY,
        'imgsz': '2mp',
        'num': min(num_images, 100)  # SerpAPI limit
    }
    
    url = 'https://serpapi.com/search?' + urlencode(params)
    
    try:
        print(f"  Searching: {query}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'images_results' in data and isinstance(data['images_results'], list):
            image_urls = []
            for img in data['images_results'][:num_images]:
                if 'original' in img:
                    image_urls.append(img['original'])
                elif 'thumbnail' in img:
                    image_urls.append(img['thumbnail'])
            
            print(f"  ✓ Found {len(image_urls)} images")
            return image_urls
        else:
            print(f"  ⚠ No images found in response")
            return []
    
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error: {e}")
        return []


def download_image(url, save_path, timeout=15):
    """
    Download an image from URL and save it.
    
    Args:
        url: Image URL
        save_path: Path to save the image
        timeout: Request timeout in seconds
    
    Returns:
        True if successful, False otherwise
    """
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check if it's actually an image
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type.lower():
            return False
        
        # Save the image
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify file was created and has content
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True
        else:
            if os.path.exists(save_path):
                os.remove(save_path)
            return False
    
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


def download_category_images(category, queries, target_count=30, split_ratio=0.8):
    """
    Download images for a specific category.
    
    Args:
        category: Category name (e.g., 'vacuum_packaged')
        queries: List of search queries for this category
        target_count: Total number of images to download
        split_ratio: Ratio for train/validation split (0.8 = 80% train, 20% val)
    """
    
    print(f"\n{'='*60}")
    print(f"Downloading images for: {category}")
    print(f"{'='*60}")
    
    # Create directories
    train_dir = Path('data/train') / category
    val_dir = Path('data/validation') / category
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate how many images per split
    train_target = int(target_count * split_ratio)
    val_target = target_count - train_target
    
    print(f"Target: {train_target} training, {val_target} validation images")
    
    all_urls = []
    
    # Search using all queries
    for query in queries:
        urls = search_images(query, num_images=15)
        all_urls.extend(urls)
        time.sleep(1)  # Rate limiting
    
    # Remove duplicates
    all_urls = list(set(all_urls))
    print(f"\nTotal unique URLs found: {len(all_urls)}")
    
    if len(all_urls) == 0:
        print("⚠ No images found for this category!")
        return
    
    # Download images
    train_count = 0
    val_count = 0
    
    for idx, url in enumerate(all_urls):
        if train_count >= train_target and val_count >= val_target:
            break
        
        # Determine if this goes to train or validation
        if train_count < train_target:
            save_dir = train_dir
            prefix = 'train'
        elif val_count < val_target:
            save_dir = val_dir
            prefix = 'val'
        else:
            break
        
        # Generate filename
        file_ext = url.split('.')[-1].split('?')[0].lower()
        if file_ext not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
            file_ext = 'jpg'
        
        filename = f"{category}_{prefix}_{idx:04d}.{file_ext}"
        save_path = save_dir / filename
        
        # Download
        print(f"  Downloading {idx+1}/{len(all_urls)}: {filename}...", end=' ')
        
        if download_image(url, save_path):
            if prefix == 'train':
                train_count += 1
            else:
                val_count += 1
            print("✓")
        else:
            print("✗ Failed")
        
        time.sleep(0.5)  # Rate limiting
    
    print(f"\n✓ Downloaded: {train_count} training, {val_count} validation images")


def main():
    """Main function to download all images."""
    
    print("\n" + "="*60)
    print("MEAT IDENTIFIER - IMAGE DOWNLOADER")
    print("="*60)
    print("\nThis script will download images for training the model.")
    print("Using SerpAPI to search Google Images.")
    
    # Ask user for confirmation
    print("\n" + "="*60)
    print("CATEGORIES TO DOWNLOAD:")
    print("="*60)
    for category, queries in SEARCH_QUERIES.items():
        print(f"\n{category}:")
        for q in queries:
            print(f"  - {q}")
    
    print("\n" + "="*60)
    
    response = input("\nDownload images? This will use your SerpAPI credits. (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("Download cancelled.")
        return
    
    # Get target count
    try:
        target = input("\nHow many images per category? (default: 30): ").strip()
        target_count = int(target) if target else 30
    except ValueError:
        target_count = 30
    
    print(f"\nDownloading {target_count} images per category...")
    
    # Download for each category
    for category, queries in SEARCH_QUERIES.items():
        download_category_images(category, queries, target_count=target_count)
    
    # Final summary
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE - SUMMARY")
    print("="*60)
    
    for split in ['train', 'validation']:
        print(f"\n{split.upper()}:")
        for category in SEARCH_QUERIES.keys():
            category_dir = Path('data') / split / category
            if category_dir.exists():
                count = len(list(category_dir.glob('*.*')))
                print(f"  {category}: {count} images")
    
    print("\n" + "="*60)
    print("Next step: Run training!")
    print("  python train.py --train-dir data/train --val-dir data/validation --epochs 20")
    print("="*60)


if __name__ == '__main__':
    main()
