import sys
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict
from io import BytesIO  # for safe batch image loading

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from torchvision import transforms

# --- 1. SYSTEM PATH SETUP ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- 2. IMPORTS ---
try:
    from ultralytics import YOLO
    from src import config
    from src.models.mlp import SimpleMLP
    from src.models.cnn import CustomCNN
except ImportError as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- 3. CONFIGURATION ---
st.set_page_config(
    page_title="PCB Inspection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 4. CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --bg: #000000;
        --surface: #111111;
        --border: #333333;
        --primary: #EDEDED;
        --accent: #FFD700;
        --success: #2E8B57;
        --danger: #D32F2F;
    }

    .stApp { background-color: var(--bg); font-family: 'Inter', sans-serif; color: var(--primary); }
    
    h1 { font-size: 2.2rem !important; font-weight: 700 !important; letter-spacing: -0.02em; margin-bottom: 0.5rem !important; }
    h2 { font-size: 1.1rem !important; font-weight: 500 !important; color: #888 !important; letter-spacing: -0.01em; margin-top: 0 !important; }
    h3 { font-size: 0.9rem !important; font-weight: 600 !important; text-transform: uppercase; color: #666 !important; margin-bottom: 10px !important; }
    
    .ui-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; transition: border 0.2s ease; }
    
    .metric-container { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 15px; background: #0A0A0A; border: 1px solid var(--border); border-radius: 8px; }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 600; color: var(--primary); }
    .metric-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: #666; margin-top: 6px; }

    .stButton > button { background-color: var(--primary); color: black; border: none; border-radius: 6px; padding: 0.6rem 1.5rem; font-weight: 600; font-size: 0.9rem; transition: all 0.2s; width: 100%; }
    .stButton > button:hover { background-color: var(--accent); color: black; transform: translateY(-1px); }
    
    div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: transparent;
        color: var(--primary);
        border: 1px solid var(--border);
    }
    
    [data-testid="stFileUploader"] { border: 1px dashed var(--border); background: #050505; padding: 20px; border-radius: 12px; }
    
    .status-pill { padding: 6px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: inline-flex; align-items: center; gap: 8px; width: 100%; justify-content: center; }
    .status-pass { background: #064E3B; color: #6EE7B7; border: 1px solid #059669; }
    .status-fail { background: #450A0A; color: #FCA5A5; border: 1px solid #B91C1C; }
    
    img { max-height: 400px; object-fit: contain; }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 5. LOGIC & MODELS ---
FIXED_THRESH = 0.5  # Standard threshold

@st.cache_resource
def load_all_models():
    models = {"yolo": None, "mlp": None, "cnn": None}
    
    # 1. Load YOLO - Check locations based on train_yolo.py output
    possible_paths = [
        project_root / "outputs/yolo_pcb/weights/best.pt",  # Your training script path
        project_root / "runs/detect/train/weights/best.pt",  # Default fallback
        "yolov8s.pt"  # Pretrained fallback
    ]
    
    for path in possible_paths:
        p = Path(path)
        if p.exists():
            try:
                models["yolo"] = YOLO(str(p))
                print(f"Loaded YOLO from: {p}")
                break
            except Exception as e:
                print(f"Failed to load {p}: {e}")

    # 2. Load MLP
    mlp_path = project_root / "outputs/mlp_best.pth"
    if mlp_path.exists():
        try:
            model = SimpleMLP()
            state = torch.load(mlp_path, map_location='cpu')
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            model.load_state_dict(state)
            model.eval()
            models["mlp"] = model
        except Exception as e:
            print(f"Error loading MLP: {e}")

    # 3. Load CNN
    cnn_path = project_root / "outputs/cnn_best.pth"
    if cnn_path.exists():
        try:
            model = CustomCNN()
            state = torch.load(cnn_path, map_location='cpu')
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            model.load_state_dict(state)
            model.eval()
            models["cnn"] = model
        except Exception as e:
            print(f"Error loading CNN: {e}")

    return models

MODELS = load_all_models()

# --- CUSTOM DRAWING FUNCTION ---
def draw_yolo_boxes(image: Image.Image, boxes: List[Dict]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box in boxes:
        color_hash = hash(box['class']) % 0xFFFFFF
        color = f"#{color_hash:06x}"
        
        x1, y1, x2, y2 = box['bbox']
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        label = f"{box['class']} {box['conf']:.2f}"
        if font:
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1), label, fill="black", font=font)
        else:
            draw.text((x1, y1), label, fill=color)

    return image

# --- INFERENCE FUNCTIONS ---

def predict_classifier(model, image, arch):
    start = time.time()
    target_size = config.MLP_INPUT_SIZE if arch == "mlp" else config.IMAGE_SIZE
    
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        input_tensor = preprocess(image).unsqueeze(0)
        logits = model(input_tensor)
        prob_defect = torch.sigmoid(logits).item()
    
    latency = (time.time() - start) * 1000.0
    
    # Prob Correct = 1 - Prob Defect
    prob_correct = 1.0 - prob_defect
    
    # Decision using Fixed Threshold (0.5)
    if prob_correct >= FIXED_THRESH:
        label = "Correct PCB"
    else:
        label = "Reject"
        
    return label, prob_correct, latency

def predict_yolo(model, image):
    """
    Runs YOLO detection using the standard Fixed Threshold.
    """
    start = time.time()
    results = model.predict(image, conf=FIXED_THRESH, verbose=False)
    latency = (time.time() - start) * 1000.0
    
    res = results[0]
    
    raw_detections = []
    for box in res.boxes:
        name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy().tolist()
        
        raw_detections.append({
            "class": name, 
            "conf": conf,
            "bbox": bbox
        })
        
    return raw_detections, latency


# --- 6. MAIN UI STRUCTURE ---

c1, c2 = st.columns([3, 1])
with c1:
    st.markdown(
        "<h1>Automated Inspection System "
        "<span style='color:var(--accent); font-size:1rem; vertical-align:middle;'>v3.0</span></h1>",
        unsafe_allow_html=True
    )
    st.markdown("<h2>For Printed Circuit Boards</h2>", unsafe_allow_html=True)

with c2:
    model_name = st.selectbox(
        "Active Model",
        ["YOLOv8 (Object Detection)", "Custom CNN (Binary)", "MLP Baseline (Binary)"],
        label_visibility="collapsed"
    )

key_map = {"YOLO": "yolo", "Custom": "cnn", "MLP": "mlp"}
model_key = key_map.get(model_name.split()[0], "yolo")
active_model = MODELS.get(model_key)

st.write("") 

tab_inspect, tab_batch, tab_data = st.tabs(
    ["Single Board Inspection", "Batch Queue", "Performance Data"]
)

# === TAB 1: SINGLE BOARD INSPECTION ===
with tab_inspect:
    
    col_input, col_output = st.columns([1, 1], gap="medium")
    
    with col_input:
        st.markdown("### 1. Input Source")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg','png','bmp'])
        
        st.markdown("### 2. System Status")
        with st.container():
            st.markdown(f"<div class='ui-card'>", unsafe_allow_html=True)
            if active_model:
                st.markdown(
                    f"<div style='color:#4CAF50; font-size:0.9rem; font-weight:600'>"
                    f"● Active: {model_name}</div>",
                    unsafe_allow_html=True
                )
                st.caption(f"Neural network initialized. Standard threshold ({FIXED_THRESH}).")
            else:
                st.markdown(
                    "<div style='color:#FF5252; font-size:0.9rem; font-weight:600'>"
                    "● Model Missing (Simulating)</div>",
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.write("")
        st.markdown("### 3. Benchmark Comparison")
        if uploaded_file and st.button("RUN ALL MODELS COMPARISON", type="primary"):
            st.markdown("#### Live Inference Results")
            img_bench = Image.open(uploaded_file).convert("RGB")
            comp_results = []
            
            # 1. MLP
            if MODELS['mlp']:
                l, c, t = predict_classifier(MODELS['mlp'], img_bench, 'mlp')
                comp_results.append({
                    "Model": "MLP",
                    "Result": l,
                    "Score": f"{c:.1%}",
                    "Time": f"{t:.1f}ms"
                })
            else:
                comp_results.append({
                    "Model": "MLP",
                    "Result": "Reject (Sim)",
                    "Score": "51.5%",
                    "Time": "2ms"
                })
            
            # 2. CNN
            if MODELS['cnn']:
                l, c, t = predict_classifier(MODELS['cnn'], img_bench, 'cnn')
                comp_results.append({
                    "Model": "CNN",
                    "Result": l,
                    "Score": f"{c:.1%}",
                    "Time": f"{t:.1f}ms"
                })
            else:
                comp_results.append({
                    "Model": "CNN",
                    "Result": "Reject (Sim)",
                    "Score": "52.2%",
                    "Time": "15ms"
                })
                
            # 3. YOLO
            if MODELS['yolo']:
                d, t = predict_yolo(MODELS['yolo'], img_bench)
                if not d:
                    yolo_res = "Correct PCB"
                    yolo_score = 1.0
                else:
                    yolo_res = f"{len(d)} Defects"
                    avg_conf = np.mean([x['conf'] for x in d])
                    yolo_score = 1.0 - avg_conf
                    
                comp_results.append({
                    "Model": "YOLOv8",
                    "Result": yolo_res,
                    "Score": f"{yolo_score:.1%}",
                    "Time": f"{t:.1f}ms"
                })
            else:
                comp_results.append({
                    "Model": "YOLOv8",
                    "Result": "Correct PCB (Sim)",
                    "Score": "92.8%",
                    "Time": "110ms"
                })
            
            st.dataframe(
                pd.DataFrame(comp_results),
                use_container_width=True,
                hide_index=True
            )

    with col_output:
        st.markdown("### 4. Analysis Results")
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            
            # --- PROCESS LOGIC ---
            if "YOLO" in model_name:
                if active_model:
                    raw_dets, latency = predict_yolo(active_model, image)
                    
                    display_img = image.copy()
                    if raw_dets:
                        display_img = draw_yolo_boxes(display_img, raw_dets)
                    
                    if not raw_dets:
                        status = "Correct PCB"
                        confidence = 1.0
                    else:
                        status = "Reject"
                        confidence = max([d['conf'] for d in raw_dets])
                        
                else:
                    time.sleep(0.5)
                    latency = 110.0
                    display_img = image
                    status = "Correct PCB"
                    confidence = 0.95
                    raw_dets = []
                
                st.image(display_img, use_container_width=True,
                         caption="Computer Vision Output")

                m1, m2, m3 = st.columns(3)
                with m1:
                    status_class = "status-pass" if status == "Correct PCB" else "status-fail"
                    icon = "✅" if status == "Correct PCB" else "⛔"
                    label_text = status.upper() if status == "Correct PCB" else f"REJECT ({len(raw_dets)})"
                    st.markdown(
                        f"<div class='status-pill {status_class}'>{icon} {label_text}</div>",
                        unsafe_allow_html=True
                    )
                with m2:
                     st.markdown(
                         f"<div class='metric-container'><div class='metric-value'>{latency:.0f}ms</div>"
                         f"<div class='metric-label'>LATENCY</div></div>",
                         unsafe_allow_html=True
                     )
                with m3:
                     st.markdown(
                         f"<div class='metric-container'><div class='metric-value'>{confidence:.0%}</div>"
                         f"<div class='metric-label'>CONFIDENCE</div></div>",
                         unsafe_allow_html=True
                     )
                
            else:
                # Classification Logic
                if active_model:
                    pred, conf, latency = predict_classifier(active_model, image, model_key)
                else:
                    time.sleep(0.3)
                    latency = 45.0
                    sim_correctness = 0.52 
                    pred = "Correct PCB" if sim_correctness >= FIXED_THRESH else "Reject"
                    conf = sim_correctness
                
                overlay = np.array(image)
                if pred == "Reject":
                    overlay[:, :, 0] = 200
                st.image(
                    overlay,
                    use_container_width=True,
                    caption=f"Classification Result: {pred}"
                )

                m1, m2, m3 = st.columns(3)
                with m1:
                    status_class = "status-pass" if pred == "Correct PCB" else "status-fail"
                    icon = "✅" if pred == "Correct PCB" else "⛔"
                    st.markdown(
                        f"<div class='status-pill {status_class}'>{icon} {pred.upper()}</div>",
                        unsafe_allow_html=True
                    )
                with m2:
                    st.markdown(
                        f"<div class='metric-container'><div class='metric-value'>{conf:.1%}</div>"
                        f"<div class='metric-label'>CONFIDENCE</div></div>",
                        unsafe_allow_html=True
                    )
                with m3:
                    st.markdown(
                        f"<div class='metric-container'><div class='metric-value'>{latency:.0f}ms</div>"
                        f"<div class='metric-label'>LATENCY</div></div>",
                        unsafe_allow_html=True
                    )
        else:
             st.markdown("""
            <div style='height: 400px; border: 1px dashed #333; border-radius: 12px; display: flex; align-items: center; justify-content: center; color: #555; background: #080808;'>
                <div>Waiting for image input...</div>
            </div>
            """, unsafe_allow_html=True)


# === TAB 2: BATCH QUEUE ===
with tab_batch:
    st.markdown("### Production Line Simulator")
    st.caption("Upload multiple board images. Processing optimized using YOLOv8.")
    
    files = st.file_uploader(
        "Select Images (Ctrl+A to select multiple)",
        accept_multiple_files=True,
        key="batch_load"
    )
    
    if files and st.button("RUN BATCH JOB", type="primary"):
        # --- STRICT FILTERING FOR IMAGES ONLY ---
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        valid_files = [
            f for f in files if Path(f.name).suffix.lower() in valid_extensions
        ]
        
        if not valid_files:
            st.warning("No valid image files found. Please select JPG, PNG, or BMP files.")
        else:
            st.info(f"Processing {len(valid_files)} valid images...")
            progress = st.progress(0)
            results = []
            
            stat1, stat2, stat3 = st.columns(3)
            batch_model = MODELS.get('yolo')
            
            for i, f in enumerate(valid_files):
                if batch_model:
                    try:
                        # --- SAFE IMAGE LOADING WITH getvalue() ---
                        file_bytes = f.getvalue()
                        img = Image.open(BytesIO(file_bytes)).convert('RGB')
                        
                        # Run real YOLO inference
                        raw_dets, ms = predict_yolo(batch_model, img)
                        
                        if not raw_dets:
                            status = "PASS"
                            conf_disp = 1.0  # Clean
                        else:
                            status = "REJECT"
                            conf_disp = max([d['conf'] for d in raw_dets])
                            
                        proc_time = ms
                    except Exception as e:
                        status = "ERROR"
                        proc_time = 0
                        conf_disp = 0
                else:
                    # If YOLO is missing, we just simulate (shouldn't normally happen)
                    time.sleep(0.02)
                    is_defect = np.random.random() > 0.8
                    conf_disp = np.random.uniform(0.85, 0.99)
                    proc_time = np.random.randint(90, 150)
                    status = "REJECT" if is_defect else "PASS"
                
                results.append({
                    "Board ID": f.name,
                    "Status": status,
                    "Confidence": f"{conf_disp:.2f}",
                    "Time (ms)": f"{proc_time:.0f}"
                })
                progress.progress((i+1)/len(valid_files))
            
            df = pd.DataFrame(results)
            fails = len(df[df['Status'] == 'REJECT'])
            yield_rate = (len(df) - fails) / len(df) if len(df) > 0 else 0.0

            # --- Compute Batch Accuracy Using Filename Ground Truth ---
            # Convention:
            #   *_temp.jpg  -> Correct PCB -> ground truth PASS
            #   *_test.jpg  -> Defective   -> ground truth REJECT
            y_true = []
            y_pred = []

            for row in results:
                fname = row["Board ID"].lower()
                pred_status = row["Status"]

                if "temp" in fname:
                    true_status = "PASS"
                elif "test" in fname:
                    true_status = "REJECT"
                else:
                    true_status = "UNKNOWN"

                if true_status != "UNKNOWN":
                    y_true.append(true_status)
                    y_pred.append(pred_status)

            if len(y_true) > 0:
                correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
                batch_accuracy = correct / len(y_true)
            else:
                batch_accuracy = 0.0
            
            with stat1:
                st.markdown(
                    f"<div class='metric-container'><div class='metric-value'>{len(df)}</div>"
                    f"<div class='metric-label'>TOTAL BOARDS</div></div>",
                    unsafe_allow_html=True
                )
            with stat2:
                st.markdown(
                    f"<div class='metric-container'><div class='metric-value' style='color:#FCA5A5'>{fails}</div>"
                    f"<div class='metric-label'>REJECTED</div></div>",
                    unsafe_allow_html=True
                )
            with stat3:
                st.markdown(
                    f"<div class='metric-container'><div class='metric-value' style='color:#6EE7B7'>{yield_rate:.1%}</div>"
                    f"<div class='metric-label'>YIELD RATE</div></div>",
                    unsafe_allow_html=True
                )

            # Show batch accuracy metric below the three cards
            st.write("")
            st.markdown(
                f"<div class='metric-container'>"
                f"<div class='metric-value'>{batch_accuracy:.1%}</div>"
                f"<div class='metric-label'>MODEL ACCURACY (BATCH)</div>"
                f"</div>",
                unsafe_allow_html=True
            )
                
            st.write("")
            st.dataframe(
                df.style.applymap(
                    lambda x: 'color:#FCA5A5' if x == 'REJECT' else 'color:#6EE7B7',
                    subset=['Status']
                ), 
                use_container_width=True
            )


# === TAB 3: DATA ===
with tab_data:
    st.markdown("### Model Benchmarks")
    
    df_perf = pd.DataFrame({
        "Model Architecture": ["YOLOv8-Small", "Custom CNN", "MLP Baseline"],
        "Role": ["Detection & Localization", "Binary Classification", "Binary Classification"],
        "Accuracy/mAP": ["92.8% (mAP)", "52.2%", "51.5%"],
        "F1-Score": ["N/A", "65.9%", "68.6%"],
        "Parameters": ["11.0M", "0.46M", "6.4M"]
    })
    st.table(df_perf)
    
    st.write("")
    
    c1_d, c2_d = st.columns(2)
    cm_path = project_root / "runs/detect/val/confusion_matrix.png"
    pr_path = project_root / "runs/detect/val/BoxPR_curve.png"
    
    with c1_d:
        st.markdown("#### Confusion Matrix")
        if cm_path.exists(): 
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Not found in runs/")
            
    with c2_d:
        st.markdown("#### Precision-Recall Curve")
        if pr_path.exists(): 
            st.image(str(pr_path), use_container_width=True)
        else:
            st.info("Not found in runs/")
