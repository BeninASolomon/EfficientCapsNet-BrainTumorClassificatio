# EfficientCapsNet for Brain Tumor Classification

This repository contains the implementation of an **Efficient Capsule Network (CapsNet)** using TensorFlow/Keras for brain tumor classification. The model classifies brain tumor MRI images into four categories: **glioma, meningioma, no tumor, and pituitary tumor**.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project implements a Capsule Network model that classifies brain MRI images into four categories: **glioma**, **meningioma**, **no tumor**, and **pituitary tumor**. The model processes grayscale images with a resolution of 128x128 pixels and uses a lightweight architecture for efficient training.

## Features
- **Efficient Capsule Network (EfficientCapsNet)** model for image classification.
- **Convolutional Layers**: Two initial convolutional layers followed by a primary capsule layer.
- **Capsule Layer**: Uses fully connected layers to classify the input images into one of four classes.
- **Sparse Categorical Crossentropy**: Loss function for multi-class classification.
- **SGD Optimizer**: Stochastic gradient descent with momentum and Nesterov accelerated gradient (NAG).

## Dataset
The dataset should be organized as follows:
```plaintext
Data/
│
├── glioma/         # Contains images of glioma brain tumors
├── meningioma/     # Contains images of meningioma brain tumors
├── notumor/        # Contains images of brain scans with no tumors
└── pituitary/      # Contains images of pituitary brain tumors
pip install tensorflow opencv-python numpy
