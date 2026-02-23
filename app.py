# app.py - Updated with novelty highlights
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from complete_pipeline import CompleteArrhythmiaPipeline
import time

st.set_page_config(page_title="Smart Arrhythmia Detection", page_icon="❤️", layout="wide")

st.title("❤️ Smart Arrhythmia Disease Detection System")
st.markdown("*Implementing novel CNN-LSTM with Signal Quality Index (SQI)*")

# Initialize pipeline
@st.cache_resource
def init_pipeline():
    return CompleteArrhythmiaPipeline('models/mitbih_cnn_lstm.h5')

pipeline = init_pipeline()

# Sidebar
with st.sidebar:
    # st.header("📋 Novelty Features")
    
    features = {
        # "✅ Image-based ECG Reconstruction": "From smartphone photos",
        # "✅ Signal Quality Index (SQI)": "Confidence scoring",
        # "✅ CNN-LSTM Architecture": "Morphological + temporal",
        # "✅ Noise-robust Processing": "Motion artifact suppression",
        # "✅ End-to-end Pipeline": "Image to diagnosis"
    }
    
    for feature, desc in features.items():
        st.success(f"{feature}\n*{desc}*")
    
    st.markdown("---")
    st.header("Input Options")
    input_type = st.radio("Select Input Type", ["📱 ECG Image", "📊 Wearable Signal"])

# Main content
if input_type == "📱 ECG Image":
    st.header("ECG Image Analysis")
    
    uploaded_file = st.file_uploader("Upload ECG Image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded ECG", use_column_width=True)
        
        if st.button("🔍 Analyze with Novel Pipeline", type="primary"):
            with st.spinner("Processing with CNN-LSTM + SQI..."):
                # Save temp file
                with open("temp.png", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Process
                start_time = time.time()
                result = pipeline.process_ecg_image("temp.png")
                proc_time = time.time() - start_time
                
                with col2:
                    st.subheader("📊 Analysis Results")
                    
                    # Main diagnosis
                    if result['reliable']:
                        st.success(f"### Diagnosis: {result['diagnosis']}")
                    else:
                        st.warning(f"### Diagnosis: {result['diagnosis']} (Low Quality)")
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Confidence", f"{result['confidence']:.1%}")
                    col_b.metric("Signal Quality", f"{result['signal_quality']:.1%}")
                    col_c.metric("Processing", f"{proc_time:.2f}s")
                    
                    # Quality metrics (NOVELTY) - FIXED HERE
                    with st.expander("📈 Signal Quality Index Details (Novel)"):
                        for metric, value in result['detailed_sqi'].items():
                            st.metric(metric.replace('_', ' ').title(), f"{value:.2f}")
                
                # Signal plot
                st.subheader("📉 Reconstructed ECG Signal")
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(result['processed_signal'], color='blue', linewidth=1)
                ax.set_xlabel("Sample")
                ax.set_ylabel("Amplitude")
                ax.set_title("Extracted Waveform with Grid Removal (Novel)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Download
                st.download_button(
                    "📥 Download Report",
                    str(result),
                    file_name="ecg_results.json"
                )