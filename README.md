# YOLO Biomedical Object Detection & Instance Segmentation

A complete pipeline for training YOLO v9+ models on custom biomedical data for object detection and instance segmentation tasks.

## 🏥 Features

- **YOLO v9+ Support**: Latest YOLO architecture with improved performance
- **Biomedical Focus**: Optimized for medical imaging and biological data
- **Instance Segmentation**: Support for both bounding box detection and pixel-level segmentation
- **Data Conversion**: Automatic conversion from various annotation formats (XML, JSON, CSV, TXT)
- **Comprehensive Training**: Full training pipeline with validation and monitoring
- **Advanced Inference**: Batch processing and real-time prediction capabilities
- **Performance Evaluation**: Detailed metrics and visualization reports

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd NewYOLO

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Organize your biomedical data in the following structure:

```
your_data/
├── image1.jpg
├── image1.xml (or .json, .txt, .csv)
├── image2.jpg
├── image2.xml
└── ...
```

Convert your annotations to YOLO format:

```bash
# Auto-detect format and convert
python data_preparation.py --input ./your_data --output ./dataset

# Specify format manually
python data_preparation.py --input ./your_data --output ./dataset --format xml

# Define custom classes
python data_preparation.py --input ./your_data --output ./dataset --classes "cell,nucleus,mitochondria"
```

### 3. Training

Start training with your prepared dataset:

```bash
# Basic training
python train.py --dataset ./dataset/dataset.yaml

# Custom training parameters
python train.py --dataset ./dataset/dataset.yaml --epochs 200 --batch-size 32 --img-size 640

# Validate dataset structure only
python train.py --dataset ./dataset --validate-only
```

### 4. Inference

Run predictions on new images:

```bash
# Single image
python inference.py --model runs/train/biomedical_yolo/exp/weights/best.pt --input ./test_image.jpg

# Batch processing
python inference.py --model runs/train/biomedical_yolo/exp/weights/best.pt --input ./test_images/ --batch

# Custom confidence threshold
python inference.py --model runs/train/biomedical_yolo/exp/weights/best.pt --input ./test_image.jpg --conf 0.5
```

### 5. Evaluation

Evaluate model performance on test data:

```bash
python evaluate.py --model runs/train/biomedical_yolo/exp/weights/best.pt --dataset ./dataset/dataset.yaml
```

## 📁 Project Structure

```
NewYOLO/
├── requirements.txt          # Python dependencies
├── config.yaml              # Training configuration
├── dataset.yaml             # Dataset configuration template
├── data_preparation.py      # Data conversion and preparation
├── train.py                 # Training script
├── inference.py             # Inference and prediction
├── evaluate.py              # Model evaluation
├── README.md                # This file
└── dataset/                 # Generated dataset directory
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── dataset.yaml
```

## ⚙️ Configuration

### Training Configuration (`config.yaml`)

Key parameters you can modify:

```yaml
# Training
epochs: 100
batch_size: 16
imgsz: 640

# Model
model: yolov9-c.pt  # yolov9-n, yolov9-s, yolov9-m, yolov9-l, yolov9-x

# Data Augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
fliplr: 0.5

# Instance Segmentation
seg: true
seg_scale: 1.0
```

### Dataset Configuration (`dataset.yaml`)

Automatically generated, but you can customize:

```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test
nc: 5  # Number of classes
names: 
  0: cell
  1: nucleus
  2: mitochondria
  3: golgi_apparatus
  4: endoplasmic_reticulum
```

## 🔧 Advanced Usage

### Custom Data Augmentation

Modify `config.yaml` for biomedical-specific augmentations:

```yaml
# Reduce aggressive augmentations for medical images
hsv_h: 0.01      # Minimal hue change
hsv_s: 0.3       # Reduced saturation change
hsv_v: 0.2       # Reduced brightness change
degrees: 0.0      # No rotation (preserve orientation)
perspective: 0.0  # No perspective distortion
```

### Multi-GPU Training

```bash
# Use multiple GPUs
python train.py --dataset ./dataset/dataset.yaml --device 0,1,2,3
```

### Transfer Learning

```bash
# Start from a custom pre-trained model
python train.py --dataset ./dataset/dataset.yaml --model ./custom_pretrained.pt
```

### Hyperparameter Tuning

```bash
# Experiment with different learning rates
python train.py --dataset ./dataset/dataset.yaml --lr0 0.001 --lrf 0.01

# Adjust loss weights
python train.py --dataset ./dataset/dataset.yaml --box 10.0 --cls 1.0 --dfl 2.0
```

## 📊 Supported Annotation Formats

### 1. XML (Pascal VOC)
```xml
<annotation>
  <object>
    <name>cell</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>200</xmax>
      <ymax>250</ymax>
    </bndbox>
  </object>
</annotation>
```

### 2. JSON (COCO/LabelMe)
```json
{
  "shapes": [
    {
      "label": "nucleus",
      "points": [[100, 150], [200, 250]]
    }
  ]
}
```

### 3. CSV
```csv
filename,class,x_min,y_min,width,height
image1.jpg,cell,100,150,100,100
```

### 4. TXT (YOLO format)
```
0 0.5 0.5 0.2 0.3
1 0.7 0.6 0.1 0.2
```

## 🎯 Biomedical Use Cases

### Cell Biology
- Cell counting and classification
- Organelle detection (nucleus, mitochondria, etc.)
- Cell cycle stage identification
- Apoptosis detection

### Medical Imaging
- Tumor detection and segmentation
- Organ identification
- Blood vessel tracking
- Tissue classification

### Microscopy
- Fluorescence image analysis
- Phase contrast microscopy
- Electron microscopy
- Time-lapse imaging

## 📈 Performance Monitoring

### Training Metrics
- Loss curves (box, class, DFL)
- mAP@0.5 and mAP@0.5:0.95
- Precision and recall
- Learning rate scheduling

### Validation Results
- Per-class performance
- Confusion matrices
- Detection examples
- Segmentation quality

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train.py --batch-size 8
   
   # Reduce image size
   python train.py --img-size 416
   ```

2. **Poor Detection Quality**
   ```bash
   # Increase training epochs
   python train.py --epochs 300
   
   # Adjust confidence threshold
   python inference.py --conf 0.1
   ```

3. **Data Format Errors**
   ```bash
   # Validate dataset structure
   python train.py --dataset ./dataset --validate-only
   ```

### Performance Tips

- Use appropriate image size (640x640 for most cases)
- Ensure balanced class distribution
- Use data augmentation sparingly for medical images
- Monitor validation metrics to prevent overfitting

## 📚 Additional Resources

- [YOLO Documentation](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Medical Image Analysis Papers](https://paperswithcode.com/task/medical-image-analysis)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Ultralytics team for YOLO implementation
- PyTorch community
- Biomedical imaging researchers worldwide

---

**Note**: This pipeline is specifically designed for biomedical applications. For general object detection tasks, consider using the standard YOLO training procedures.
