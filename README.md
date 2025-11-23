# Meat Identifier AI 🥩

An AI-powered computer vision system built with Python to identify raw meat (including game, meat, and poultry) in various packaging states:
- Vacuum-sealed packages
- Aluminum foil wrapped
- Frozen state

## Features

- **Deep Learning Model**: Uses MobileNetV2 with transfer learning for efficient and accurate classification
- **Multiple Packaging Types**: Identifies meat in vacuum packaging, aluminum foil, or frozen state
- **Easy to Use**: Simple command-line interface for both single images and batch processing
- **Extensible**: Easy to retrain on custom datasets
- **Feature Analysis**: Optional packaging feature detection to help understand predictions

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Quick Start

### Download images

```bash
python download_images.py
```

## Training Your Own Model

To train the model on your custom dataset:

### 1. Prepare Your Dataset

Organize your images in the following structure:

```
dataset/
├── train/
│   ├── no_meat/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── vacuum_packaged/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── aluminum_foil/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── frozen/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── validation/
    ├── no_meat/
    ├── vacuum_packaged/
    ├── aluminum_foil/
    └── frozen/
```

### 2. Run Training

```bash
python train.py --train-dir dataset/train --val-dir dataset/validation --epochs 20
```

Advanced options:

```bash
python train.py \
  --train-dir dataset/train \
  --val-dir dataset/validation \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 0.0001
```

The trained model will be saved to:
- `models/meat_identifier.h5` (default model)
- `models/meat_identifier_TIMESTAMP.h5` (timestamped backup)
- `models/training_history_TIMESTAMP.json` (training metrics)

## Usage

### Option 1: Build an Untrained Model

If you want to create the model structure (for testing or before training):

```bash
python predict.py --build-model test_image.jpg
```

This creates a new untrained model at `models/meat_identifier.h5`.

### Option 2: Use a Pre-trained Model

If you have a trained model file, place it in the `models/` directory and use it for predictions.

### Single Image Prediction

Classify a single image:

```bash
python predict.py path/to/image.jpg --model models/meat_identifier.h5
```

With feature analysis:

```bash
python predict.py path/to/image.jpg --model models/meat_identifier.h5 --features
```

### Batch Processing

Classify all images in a directory:

```bash
python predict.py path/to/image_folder/ --model models/meat_identifier.h5
```

Save results to a JSON file:

```bash
python predict.py path/to/image_folder/ --model models/meat_identifier.h5 --output results.json
```

## Model Architecture

The model uses **MobileNetV2** as a base with:
- Transfer learning from ImageNet weights
- Custom classification head with:
  - Global Average Pooling
  - Dense layer (512 units) with ReLU
  - Dropout (0.5)
  - Dense layer (256 units) with ReLU
  - Dropout (0.3)
  - Output layer (4 classes) with Softmax

### Classes
1. **No Meat** - Images without meat
2. **Vacuum Packaged Meat** - Meat in transparent vacuum-sealed packages
3. **Aluminum Foil Packaged Meat** - Meat wrapped in aluminum foil
4. **Frozen Meat** - Meat in frozen state (with ice crystals, frost)

## Project Structure

```
pythonMeatIdentifier/
├── meat/                      # Core package
│   ├── __init__.py
│   ├── model.py              # Model architecture and training
│   └── preprocessing.py      # Image preprocessing utilities
├── predict.py                # Main prediction script
├── train.py                  # Training script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── models/                   # Saved models (created automatically)
```

## How It Works

1. **Image Loading**: Images are loaded and resized to 224x224 pixels
2. **Preprocessing**: Images are normalized to 0-1 range
3. **Feature Extraction**: MobileNetV2 extracts visual features
4. **Classification**: Custom layers classify the packaging type
5. **Output**: Returns class prediction with confidence scores

### Optional Feature Analysis

The system can also analyze visual features that indicate packaging type:
- **Brightness variance** (high for reflective aluminum foil)
- **Edge variance** (high for frozen meat with ice crystals)
- **Color analysis** (red/brown hues indicate raw meat)
- **Saturation** (helps identify vacuum packaging)

## Example Output

```
Analyzing: sample_images/vacuum_meat.jpg
Prediction: Vacuum Packaged Meat
Confidence: 94.23%

All probabilities:
  No Meat: 1.23%
  Vacuum Packaged Meat: 94.23%
  Aluminum Foil Packaged Meat: 3.12%
  Frozen Meat: 1.42%
```

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- OpenCV 4.8+
- NumPy 1.24+
- Pillow 10.0+
- Matplotlib 3.7+
- scikit-learn 1.3+

## Tips for Best Results

1. **Good lighting**: Ensure images are well-lit
2. **Clear visibility**: The meat/package should be clearly visible
3. **Variety in training**: Include diverse examples in your training dataset
4. **Sufficient data**: Use at least 100-200 images per class for training
5. **Validation split**: Keep 20% of data for validation

## Limitations

- Model accuracy depends on training data quality
- May struggle with unusual packaging types not seen during training
- Performance may vary with very poor lighting or obscured views
- Requires retraining for additional packaging types

## Future Improvements

- [ ] Add support for more packaging types (paper wrap, butcher paper, etc.)
- [ ] Implement real-time video stream processing
- [ ] Add meat type classification (beef, pork, chicken, etc.)
- [ ] Web interface for easier usage
- [ ] Mobile app deployment
- [ ] Quality assessment (freshness indicators)

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## Troubleshooting

### Import errors for TensorFlow/OpenCV
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Model file not found
Build a new model or ensure the model path is correct:
```bash
python predict.py --build-model dummy.jpg
```

### Out of memory during training
Reduce batch size:
```bash
python train.py --train-dir data/train --val-dir data/val --batch-size 16
```

### Poor prediction accuracy
- Ensure sufficient training data (100+ images per class)
- Train for more epochs
- Check that your validation data is representative
- Verify image quality and variety

---

**Built with ❤️ using Python, TensorFlow, and OpenCV**
