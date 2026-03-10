# -*- coding: utf-8 -*-
"""
Crop Yield Prediction - Single File Streamlit Application
Model Path: C:/Users/balasubramaniam/model.pth
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import re
import zipfile
import tempfile
import matplotlib.pyplot as plt
from io import BytesIO

# Try to import rasterio
try:
    import rasterio
except ImportError:
    st.error("Please install rasterio: pip install rasterio")
    st.stop()

# ============================================
# CONFIGURATION - YOUR MODEL PATH
# ============================================
MODEL_PATH = "C:/Users/balasubramaniam/model.pth"

# ============================================
# MODEL ARCHITECTURE
# ============================================

class HistogramCNN_LSTM(nn.Module):
    """
    Deep Learning Model for Crop Yield Prediction
    CNN-LSTM with Attention Mechanism
    """

    def __init__(self, n_bands=9, n_bins=32, n_timesteps=32,
                 cnn_channels=64, lstm_hidden=128, dropout=0.3):
        super().__init__()

        # CNN Component
        self.histogram_cnn = nn.Sequential(
            nn.Conv1d(n_bands, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # LSTM Component
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Attention Component
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1)
        )

        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size, n_bands, n_bins, n_timesteps = x.shape

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size * n_timesteps, n_bands, n_bins)

        cnn_out = self.histogram_cnn(x)
        cnn_out = cnn_out.squeeze(-1).view(batch_size, n_timesteps, -1)

        lstm_out, _ = self.lstm(cnn_out)

        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)

        output = self.regressor(context)
        return output.squeeze(-1)


# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_frame_number(filepath):
    """Extract frame number from filename"""
    filename = os.path.basename(filepath)
    name = filename.replace('.tif', '').replace('.TIF', '')
    numbers = re.findall(r'\d+', name)
    if numbers:
        return int(numbers[-1])
    return 0


def extract_zip(zip_file, extract_path):
    """Extract ZIP file and return list of TIF files"""
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    tif_files = []
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                tif_files.append(os.path.join(root, file))
    
    return sorted(tif_files, key=get_frame_number)


def geotiff_to_histogram(filepath, n_bins=32):
    """Convert GeoTIFF to histogram"""
    with rasterio.open(filepath) as src:
        bands = src.read().astype(np.float32)

    bands = bands[:7] if bands.shape[0] >= 7 else bands
    bands = bands / 10000.0
    bands = np.clip(bands, 0, 1)

    if bands.shape[0] < 4:
        raise ValueError(f"Expected at least 4 bands, got {bands.shape[0]}")

    red = bands[0]
    nir = bands[1]
    blue = bands[2]
    green = bands[3]
    swir1 = bands[4] if bands.shape[0] > 4 else np.zeros_like(red)
    swir2 = bands[5] if bands.shape[0] > 5 else np.zeros_like(red)
    swir3 = bands[6] if bands.shape[0] > 6 else np.zeros_like(red)

    ndvi = (nir - red) / (nir + red + 1e-10)
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-10)
    ndvi = np.clip(ndvi, 0, 1)
    evi = np.clip(evi, 0, 1)

    histograms = np.zeros((9, n_bins), dtype=np.float32)
    features = [red, nir, blue, green, swir1, swir2, swir3, ndvi, evi]

    for i, feature in enumerate(features):
        hist, _ = np.histogram(feature.flatten(), bins=n_bins, range=(0, 1))
        hist_sum = hist.sum()
        if hist_sum > 0:
            histograms[i] = hist.astype(np.float32) / hist_sum

    return histograms


def process_satellite_images(tif_files, n_bins=32, target_timesteps=32):
    """Process all satellite images and create histogram features"""
    all_histograms = []
    ndvi_means = []
    bin_centers = np.linspace(0, 1, n_bins)
    processing_info = []
    
    for filepath in tif_files:
        frame_num = get_frame_number(filepath)
        hist = geotiff_to_histogram(filepath, n_bins)
        all_histograms.append(hist)
        
        ndvi_mean = np.sum(hist[7] * bin_centers)
        ndvi_means.append(ndvi_mean)
        
        processing_info.append({
            'frame': frame_num,
            'filename': os.path.basename(filepath),
            'ndvi': ndvi_mean
        })
    
    all_histograms = np.array(all_histograms)
    
    if all_histograms.shape[0] < target_timesteps:
        pad_count = target_timesteps - all_histograms.shape[0]
        padding = np.repeat(all_histograms[-1:], pad_count, axis=0)
        all_histograms = np.concatenate([all_histograms, padding], axis=0)
    elif all_histograms.shape[0] > target_timesteps:
        indices = np.linspace(0, all_histograms.shape[0] - 1, target_timesteps).astype(int)
        all_histograms = all_histograms[indices]
    
    histogram_features = np.transpose(all_histograms, (1, 2, 0))
    
    return histogram_features, ndvi_means, processing_info


def load_image_data(filepath):
    """Load and process a single GeoTIFF for visualization"""
    with rasterio.open(filepath) as src:
        bands = src.read().astype(np.float32)
    
    bands = bands / 10000.0
    bands = np.clip(bands, 0, 1)
    
    red = bands[0]
    nir = bands[1]
    blue = bands[2]
    green = bands[3]
    
    ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi = np.clip(ndvi, -1, 1)
    
    rgb = np.stack([
        np.clip(red * 3, 0, 1),
        np.clip(green * 3, 0, 1),
        np.clip(blue * 3, 0, 1)
    ], axis=-1)
    
    return rgb, ndvi


def create_visualization(tif_files):
    """Create visualization of satellite images and NDVI"""
    n_frames = len(tif_files)
    
    if n_frames >= 5:
        sample_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
    else:
        sample_indices = list(range(n_frames))
    
    n_samples = len(sample_indices)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))
    
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    for idx, frame_idx in enumerate(sample_indices):
        filepath = tif_files[frame_idx]
        frame_num = get_frame_number(filepath)
        
        rgb, ndvi = load_image_data(filepath)
        
        axes[0, idx].imshow(rgb)
        axes[0, idx].set_title(f'Frame {frame_num}\n(True Color)', fontsize=11, fontweight='bold')
        axes[0, idx].axis('off')
        
        im = axes[1, idx].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=1.0)
        axes[1, idx].set_title(f'Frame {frame_num}\n(NDVI: {np.mean(ndvi):.3f})', fontsize=11, fontweight='bold')
        axes[1, idx].axis('off')
    
    cbar = fig.colorbar(im, ax=axes[1, :], orientation='horizontal',
                        fraction=0.05, pad=0.08, aspect=50)
    cbar.set_label('NDVI (Vegetation Index)', fontsize=12, fontweight='bold')
    
    plt.suptitle('Satellite Images - Growing Season Progression\n' +
                 'Top: True Color (RGB) | Bottom: Vegetation Index (NDVI)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


def create_ndvi_timeline(ndvi_means, processing_info):
    """Create NDVI timeline chart"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    frames = [info['frame'] for info in processing_info]
    
    ax.plot(frames, ndvi_means, 'g-o', linewidth=2, markersize=8, label='NDVI')
    ax.fill_between(frames, ndvi_means, alpha=0.3, color='green')
    
    max_idx = np.argmax(ndvi_means)
    ax.annotate(f'Peak: {ndvi_means[max_idx]:.3f}',
                xy=(frames[max_idx], ndvi_means[max_idx]),
                xytext=(frames[max_idx], ndvi_means[max_idx] + 0.05),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='darkgreen'))
    
    ax.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean NDVI', fontsize=12, fontweight='bold')
    ax.set_title('Vegetation Index (NDVI) Throughout Growing Season', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(ndvi_means) + 0.1)
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


# ============================================
# MODEL LOADING
# ============================================

@st.cache_resource
def load_prediction_model():
    """Load the trained model (cached)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(MODEL_PATH):
        return None, None, device
    
    try:
        model = HistogramCNN_LSTM().to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, device


def make_prediction(model, histogram_features, device):
    """Make yield prediction using the model"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    model.eval()
    
    with torch.no_grad():
        input_tensor = torch.FloatTensor(histogram_features).unsqueeze(0).to(device)
        raw_output = model(input_tensor).item()
    
    # Denormalize
    label_mean = 3.2
    label_std = 0.8
    model_output = (raw_output * label_std) + label_mean
    
    # Keep prediction realistic
    final_yield = np.clip(model_output, 2.5, 4.5)
    
    return final_yield, raw_output


# ============================================
# DISPLAY FUNCTIONS
# ============================================

def display_results(predicted_yield, raw_output, ndvi_means, tif_files):
    """Display prediction results"""
    
    st.markdown("---")
    st.markdown("## IOWA 2024 SOYBEAN YIELD PREDICTION")
    st.markdown("=" * 70)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Yield", f"{predicted_yield:.2f} t/ha")
    
    with col2:
        bushels_acre = predicted_yield / 0.0672
        st.metric("In Bushels/Acre", f"{bushels_acre:.1f} bu/acre")
    
    with col3:
        iowa_avg = 3.67
        diff = predicted_yield - iowa_avg
        st.metric("vs Iowa 5-yr Avg", f"{diff:+.2f} t/ha")
    
    # Comparison text (same as notebook)
    st.markdown("---")
    st.markdown("### Comparison")
    
    comparison_text = f"""
   Comparison:
   |-- Iowa 5-year avg:  3.67 t/ha
   |-- Your prediction:  {predicted_yield:.2f} t/ha
   |-- Difference:       {predicted_yield - 3.67:+.2f} t/ha

   Data Quality:
   |-- Peak NDVI:  {max(ndvi_means):.3f}
   |-- Images:     {len(tif_files)}
    """
    st.code(comparison_text, language=None)
    
    # Interpretation
    st.markdown("---")
    st.markdown("### Interpretation")
    
    if predicted_yield > iowa_avg:
        st.success(f"""
        **Above Average Yield Predicted**
        
        The predicted yield of **{predicted_yield:.2f} t/ha** is **{diff:+.2f} t/ha** above the Iowa 5-year average.
        This suggests favorable growing conditions.
        """)
    elif predicted_yield < iowa_avg - 0.3:
        st.warning(f"""
        **Below Average Yield Predicted**
        
        The predicted yield of **{predicted_yield:.2f} t/ha** is **{diff:.2f} t/ha** below the Iowa 5-year average.
        This may indicate stress factors during the growing season.
        """)
    else:
        st.info(f"""
        **Average Yield Predicted**
        
        The predicted yield of **{predicted_yield:.2f} t/ha** is close to the Iowa 5-year average of 3.67 t/ha.
        """)


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    # Page config
    st.set_page_config(
        page_title="Crop Yield Prediction",
        page_icon="🌾",
        layout="wide"
    )
    
    # Header
    st.title("🌾 Crop Yield Prediction")
    st.markdown("*Using Satellite Imagery and Deep Learning*")
    
    # Sidebar
    with st.sidebar:
        st.title("About")
        st.markdown("""
        This application predicts **soybean crop yield** using:
        - Satellite imagery (MODIS)
        - CNN-LSTM Deep Learning
        - Vegetation indices (NDVI, EVI)
        """)
        
        st.markdown("---")
        st.markdown("### Model Location")
        st.code(MODEL_PATH, language=None)
        
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        1. Upload ZIP file with satellite images
        2. Wait for processing
        3. View visualizations
        4. Get yield prediction
        """)
    
    # Load model
    model, checkpoint, device = load_prediction_model()
    
    # Model status
    st.markdown("### System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if model is not None:
            st.success("Model Loaded")
        else:
            st.error("Model Not Found")
    
    with col2:
        device_name = "GPU" if device.type == 'cuda' else "CPU"
        st.info(f"Device: {device_name}")
    
    with col3:
        if checkpoint:
            epoch = checkpoint.get('epoch', -1) + 1
            st.info(f"Epoch: {epoch}")
    
    if model is not None and checkpoint:
        dev_r2 = checkpoint.get('dev_r2', 0)
        dev_rmse = checkpoint.get('dev_rmse', 0)
        st.markdown(f"**Model Info:** Validation R2: {dev_r2:.4f} | RMSE: {dev_rmse:.4f}")
    
    st.markdown("---")
    
    # File upload
    st.header("Upload Satellite Data")
    
    uploaded_file = st.file_uploader(
        "Upload ZIP file containing satellite images (GeoTIFF format)",
        type=['zip']
    )
    
    if uploaded_file is not None and model is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save ZIP
            zip_path = os.path.join(temp_dir, 'uploaded.zip')
            with open(zip_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            # Extract
            st.markdown("### Extracting Files")
            with st.spinner("Extracting ZIP file..."):
                try:
                    tif_files = extract_zip(zip_path, temp_dir)
                    st.success(f"Found {len(tif_files)} satellite images")
                except Exception as e:
                    st.error(f"Error extracting files: {str(e)}")
                    return
            
            if len(tif_files) == 0:
                st.error("No GeoTIFF files found in the uploaded ZIP")
                return
            
            # Show file list
            with st.expander("View Uploaded Files", expanded=False):
                for i, f in enumerate(tif_files):
                    frame = get_frame_number(f)
                    st.text(f"   Frame {frame:2d} | {os.path.basename(f)}")
            
            # Process images
            st.markdown("### Processing Images")
            
            with st.spinner("Creating histogram features..."):
                try:
                    histogram_features, ndvi_means, processing_info = process_satellite_images(tif_files)
                    st.success("Image processing complete!")
                except Exception as e:
                    st.error(f"Error processing images: {str(e)}")
                    return
            
            # Info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Frames", len(tif_files))
            with col2:
                st.metric("Peak NDVI", f"{max(ndvi_means):.3f}")
            with col3:
                st.metric("Feature Shape", str(histogram_features.shape))
            
            # NDVI details
            with st.expander("View NDVI Processing Details", expanded=False):
                for info in processing_info:
                    st.text(f"   Frame {info['frame']:2d} -> NDVI: {info['ndvi']:.3f}")
            
            # Visualizations
            st.markdown("---")
            st.header("Visualizations")
            
            tab1, tab2 = st.tabs(["Satellite Images", "NDVI Timeline"])
            
            with tab1:
                with st.spinner("Generating visualization..."):
                    try:
                        vis_buf = create_visualization(tif_files)
                        st.image(vis_buf, caption="Satellite Images Throughout Growing Season", 
                                use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            with tab2:
                with st.spinner("Generating timeline..."):
                    try:
                        ndvi_buf = create_ndvi_timeline(ndvi_means, processing_info)
                        st.image(ndvi_buf, caption="NDVI Changes Throughout Growing Season", 
                                use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            # Prediction
            st.markdown("---")
            st.header("Make Prediction")
            
            if st.button("PREDICT YIELD", type="primary", use_container_width=True):
                with st.spinner("Making prediction..."):
                    try:
                        predicted_yield, raw_output = make_prediction(
                            model, histogram_features, device
                        )
                        display_results(predicted_yield, raw_output, ndvi_means, tif_files)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.exception(e)
    
    elif uploaded_file is None:
        st.info("Please upload a ZIP file containing satellite imagery to begin.")
        
        st.markdown("---")
        st.markdown("### How This Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 1. Data Input
            Upload MODIS satellite imagery.
            """)
        
        with col2:
            st.markdown("""
            #### 2. Feature Extraction
            Convert to histogram features.
            """)
        
        with col3:
            st.markdown("""
            #### 3. Prediction
            CNN-LSTM predicts yield.
            """)
    
    elif model is None:
        st.error(f"""
        **Model Not Found**
        
        Please ensure the model file exists at:
        {MODEL_PATH}
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Crop Yield Prediction System | Built with Streamlit & PyTorch*")


if __name__ == "__main__":
    main()
