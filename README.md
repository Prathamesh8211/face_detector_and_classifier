# face_detector_and_classifier

This project implements a complete **Machine Learning pipeline** that:
1. Detects faces from **static images and video streams**
2. Saves cropped face images
3. Trains a **lightweight CNN classifier**
4. Classifies faces into four categories:
   - Neutral Face
   - Face with Mask
   - Face with Sunglasses
   - Face with Mask + Sunglasses

The project uses **MediaPipe** for face detection and **MobileNetV3-Small (PyTorch)** for classification.

---

## Project Structure


face_detector_and_classifier/
├── data/
│ ├── raw/ # Raw input images (4 classes)
│ ├── cropped/ # Cropped faces from static images
│ ├── videos/ # Input video files
│ ├── video_crops/ # Cropped faces from videos
│ └── dataset/ # Train / Val / Test split
├── detector/
│ ├── detect_images.py # Face detection from images
│ └── detect_video.py # Face detection from video
├── classifier/
│ ├── split_dataset.py # Dataset splitting script
│ ├── train.py # Model training script
│ └── evaluate.py # Evaluation & confusion matrix
├── models/
│ └── face_classifier.pth # Trained model weights
├── results/
│ └── confusion_matrix.png # Confusion matrix plot

Data Preparation
Face Categories

The dataset contains four classes:

neutral

mask

sunglasses

mask_sunglasses

Running the Face Detector

1. Detect Faces from Images
   
python detector/detect_images.py

Dataset Preparation (Train / Val / Test Split)

python classifier/split_dataset.py


Creates:

data/dataset/
├── train/
├── val/
└── test/

Training the Classifier

python classifier/train.py


Model used: MobileNetV3-Small

Framework: PyTorch

Output model saved to:

models/face_classifier.pth

Evaluation & Confusion Matrix

python classifier/evaluate.py


Outputs:

Test Accuracy (printed in terminal)

Confusion Matrix saved as:

results/confusion_matrix.png
