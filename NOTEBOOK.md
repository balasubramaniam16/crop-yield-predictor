---

## 📓 Training Notebook

### 🚀 View the Complete Training Process

The Jupyter notebook contains the full model development pipeline, from data loading to model export.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DKYsCyy589m4omU4hr5Wf6OXXGQYE0qa?usp=sharing)

**Direct Download:** [Download .ipynb file](https://drive.google.com/uc?export=download&id=1DKYsCyy589m4omU4hr5Wf6OXXGQYE0qa)

### 📚 Notebook Contents

| Section | Description |
|---------|-------------|
| 📦 **Dataset Download** | Automatic download of SustainBench Crop Yield Dataset |
| 🔍 **Data Exploration** | Visualize satellite imagery and histogram features |
| 🏗️ **Model Architecture** | CNN-LSTM with attention mechanism implementation |
| 🎯 **Training Loop** | 40 epochs with validation monitoring and early stopping |
| 📊 **Evaluation** | Performance metrics (R², RMSE, MAE) on test set |
| 🖼️ **Visualizations** | Satellite images, NDVI timeline, prediction results |
| 💾 **Model Export** | Save trained model as `best_sustainbench_model.pth` |

### 🎮 How to Use the Notebook

**Option 1: Google Colab (Recommended - No Installation Required)**

1. Click the "Open in Colab" badge above
2. **Runtime** → **Change runtime type** → Select **GPU**
3. **Runtime** → **Run all**
4. Wait for training to complete (~2-3 hours)
5. Download the trained model from the final cell

**Option 2: Local Jupyter Notebook**

1. Download the notebook using the link above
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
