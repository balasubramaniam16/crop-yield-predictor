# 🌾 Crop Yield Predictor

Deep learning application for predicting soybean crop yield using satellite imagery and CNN-LSTM neural networks.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Overview

This application uses satellite imagery to predict crop yields using artificial intelligence. It analyzes vegetation health throughout the growing season to forecast soybean productivity.

### Key Features

- 🛰️ **Satellite Image Processing**: Processes MODIS satellite imagery (GeoTIFF format)
- 🧠 **Deep Learning Model**: CNN-LSTM architecture with attention mechanism
- 📊 **Vegetation Analysis**: Tracks NDVI and EVI throughout growing season
- 📈 **Interactive Visualizations**: Real-time satellite images and vegetation trends
- 🎯 **Accurate Predictions**: Trained on real agricultural data (R² = 0.81)

## 🏗️ Model Architecture
Input: Satellite Time Series (9 bands × 32 bins × 32 timesteps)

↓

[CNN] → Extract spatial features from histogram distributions

↓

[Bidirectional LSTM] → Capture temporal patterns (2 layers, 128 units)

↓

[Attention] → Focus on critical growth periods

↓

[Regression] → Output yield prediction (tons/hectare)

### Architecture Components

1. **CNN**: Processes histogram distributions of spectral bands
2. **Bidirectional LSTM**: Learns temporal dependencies (2 layers, 128 hidden units)
3. **Attention Mechanism**: Identifies important timesteps (flowering, pod fill)
4. **Regression Head**: 3-layer MLP for final yield prediction

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- GPU optional (CPU supported)

### Setup



```bash
git clone https://github.com/balasubramaniam16/crop-yield-predictor.git
cd crop-yield-predictor
Create virtual environment (recommended)
Bash

# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
Install dependencies
Bash

pip install -r requirements.txt
