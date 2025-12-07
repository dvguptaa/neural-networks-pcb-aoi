# ==========================================
# Title:  app.py - PCB Defect Detection Web UI
# Author: ECE539 Group 22
# Date:   Nov 2025
# ==========================================

"""
Streamlit web application for PCB defect detection using YOLO.
Factory operator uploads PCB images and visualizes detected defects.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import streamlit as st
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from ultralytics import YOLO
except ImportError:
    st.error("❌ ultralytics not installed! Run: pip install ultralytics")
    st.stop()


def find_model_weights() -> Optional[Path]:
    """
    Automatically search for YOLO model weights (.pt files).
    Prioritizes files with 'best' in the name.
    
    Returns:
        Path to the best model weights file, or None if not found.
    """
    # Search directories
    search_dirs = [
        project_root / "outputs" / "yolo_pcb" / "weights",
        project_root / "outputs",
        project_root / "runs",
        project_root,
    ]
    
    pt_files = []
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        # Find all .pt files recursively
        for pt_file in search_dir.rglob("*.pt"):
            pt_files.append(pt_file)
    
    if not pt_files:
        return None
    
    # Prioritize files with 'best' in the name
    best_files = [f for f in pt_files if 'best' in f.name.lower()]
    if best_files:
        # Return the first 'best' file found
        return best_files[0]
    
    # If no 'best' file, return the first .pt file found
    return pt_files[0]


@st.cache_resource
def load_model(weights_path: Path):
    """
    Load YOLO model from weights file.
    Cached to avoid reloading on every interaction.
    
    Args:
        weights_path: Path to the .pt weights file
        
    Returns:
        Loaded YOLO model
    """
    try:
        model = YOLO(str(weights_path))
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


def run_inference(model, image: Image.Image, conf_threshold: float = 0.25) -> Tuple[np.ndarray, int, list]:
    """
    Run YOLO inference on uploaded image.
    
    Args:
        model: YOLO model
        image: PIL Image
        conf_threshold: Confidence threshold for detections
        
    Returns:
        Tuple of (annotated image array, defect count, detection list)
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Run inference
    results = model.predict(
        img_array,
        conf=conf_threshold,
        verbose=False,
        save=False
    )
    
    # Get first result (single image)
    result = results[0]
    
    # Get annotated image
    annotated_img = result.plot()
    
    # Count defects
    defect_count = len(result.boxes) if result.boxes is not None else 0
    
    # Extract detection details
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls] if cls < len(model.names) else f"class_{cls}"
            detections.append({
                'class': class_name,
                'confidence': conf,
                'bbox': box.xyxy[0].cpu().numpy().tolist()
            })
    
    return annotated_img, defect_count, detections


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="PCB Defect Detection",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title
    st.title("🔍 PCB Defect Detection System")
    st.markdown("**Industrial Production AOI (Automated Optical Inspection)**")
    st.markdown("---")
    
    # Sidebar - Model Loading
    st.sidebar.header("⚙️ Model Configuration")
    
    # Find and load model
    weights_path = find_model_weights()
    
    if weights_path is None:
        st.sidebar.error("❌ No YOLO model weights (.pt) found!")
        st.sidebar.info("Please train the model first:\n```bash\npython -m src.train_yolo --train\n```")
        st.stop()
    
    st.sidebar.success(f"✅ Model found: `{weights_path.name}`")
    st.sidebar.info(f"📍 Path: `{weights_path.parent}`")
    
    # Load model
    with st.spinner("Loading YOLO model..."):
        model = load_model(weights_path)
    
    if model is None:
        st.error("Failed to load model. Please check the weights file.")
        st.stop()
    
    st.sidebar.success("✅ Model loaded successfully!")
    
    # Sidebar - Detection Settings
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Detection Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for defect detection"
    )
    
    # Main content area
    st.sidebar.markdown("---")
    st.sidebar.header("📤 Image Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload PCB Image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image of a PCB for defect detection"
    )
    
    # Initialize session state for results
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    
    # Main display area
    if uploaded_file is not None:
        # Load and display original image
        try:
            image = Image.open(uploaded_file)
            st.session_state.original_image = image
            
            # Analyze button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_button = st.button(
                    "🔍 Analyze PCB",
                    type="primary",
                    use_container_width=True
                )
            
            if analyze_button or st.session_state.detection_results is not None:
                # Run inference
                with st.spinner("Running defect detection..."):
                    annotated_img, defect_count, detections = run_inference(
                        model, image, conf_threshold
                    )
                    st.session_state.detection_results = {
                        'annotated': annotated_img,
                        'count': defect_count,
                        'detections': detections
                    }
            
            # Display results
            if st.session_state.detection_results is not None:
                results = st.session_state.detection_results
                
                # Sidebar metrics
                st.sidebar.markdown("---")
                st.sidebar.header("📊 Detection Results")
                st.sidebar.metric("Defects Found", results['count'])
                
                if results['count'] > 0:
                    st.sidebar.markdown("**Defect Details:**")
                    for i, det in enumerate(results['detections'], 1):
                        st.sidebar.markdown(
                            f"**{i}. {det['class'].upper()}**\n"
                            f"   Confidence: {det['confidence']:.2%}"
                        )
                else:
                    st.sidebar.success("✅ No defects detected!")
                
                # Side-by-side display
                st.markdown("### 📸 Detection Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Image**")
                    st.image(image, use_container_width=True, channels="RGB")
                
                with col2:
                    st.markdown("**Detected Defects**")
                    st.image(
                        results['annotated'],
                        use_container_width=True,
                        channels="RGB"
                    )
                
                # Summary
                st.markdown("---")
                if results['count'] > 0:
                    st.warning(f"⚠️ **{results['count']} defect(s) detected!** Please review the annotated image.")
                else:
                    st.success("✅ **No defects detected.** PCB passes inspection.")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            st.info("Please upload a valid image file.")
    
    else:
        # Welcome message
        st.info("👆 **Please upload a PCB image using the sidebar to begin.**")
        
        # Example usage
        st.markdown("### 📖 Usage Instructions")
        st.markdown("""
        1. **Upload Image**: Use the file uploader in the sidebar to select a PCB image
        2. **Adjust Settings**: Optionally adjust the confidence threshold
        3. **Analyze**: Click the "Analyze PCB" button to run defect detection
        4. **Review Results**: View the side-by-side comparison and defect count
        """)


if __name__ == "__main__":
    main()

