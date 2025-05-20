---
title: Scene Classification Model
emoji: 🏞️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
---

# Scene Classification Model

This project implements a scene classification model using EfficientNet-B0. The model is trained to classify images into six different categories:
- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

## Dataset
https://www.kaggle.com/datasets/puneet6060/intel-image-classification

## Model Details
- Base Model: EfficientNet-B0
- Training: Fine-tuned on the last 3 blocks
- Input Size: 224x224
- Output: 6 classes

## Training Process
- Batch Size: 16
- Epochs: 3
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Cross Entropy Loss

## Usage
The model is saved as `scene_classifier.pkl` and can be used for inference on new images.

## How to Use

1. Upload an image using the interface
2. The model will predict the scene category and show confidence scores for each class
3. The prediction with the highest confidence score will be the final classification

## Technical Details

The model was trained using:
- Transfer learning with EfficientNet-B0
- Adam optimizer
- Cross-entropy loss
- Data augmentation with resizing and normalization

## Files

- `train.py`: Script for training the model
- `app.py`: Gradio interface for the model
- `scene_classifier.pkl`: Trained model weights
- `requirements.txt`: Required Python packages 
