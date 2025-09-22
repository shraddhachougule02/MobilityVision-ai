import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile

# -----------------------------
# Load Models
# -----------------------------
people_model = YOLO("../best2.pt")
vehicles_model = YOLO("../best.pt")

# -----------------------------
# Tracking Helper
# -----------------------------
BYTE_TRACK_YAML = """
tracker_type: bytetrack
track_high_thresh: 0.35
track_low_thresh: 0.10
new_track_thresh: 0.35
track_buffer: 40
match_thresh: 0.8
conf_thres: 0.1
fuse_score: True
"""

def write_tracker_yaml() -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    f.write(BYTE_TRACK_YAML.encode("utf-8"))
    f.flush()
    f.close()
    return f.name

def extract_ids(boxes):
    if boxes is None or boxes.id is None:
        return []
    return [int(x) for x in boxes.id.int().cpu().tolist()]

# -----------------------------
# Session State
# -----------------------------
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False
if "unique_people_ids" not in st.session_state:
    st.session_state.unique_people_ids = set()
if "unique_vehicle_ids" not in st.session_state:
    st.session_state.unique_vehicle_ids = set()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="MobilityVision", layout="wide")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
/* General App */
.stApp {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    font-family: 'Segoe UI', sans-serif;
}

/* Header */
.header {
    text-align: center;
    padding: 1rem 0;
    margin-bottom: 1rem;
    background: linear-gradient(90deg, #d97706, #facc15, #6366f1, #ec4899);
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.1);
}
.header h1 {
    font-size: 3rem;
    font-weight: 900;
    margin: 0;
    color: white;
}
.header p {
    font-size: 1.2rem;
    font-style: italic;
    margin: 0;
    color: #fef9c3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    padding: 1rem;
}
[data-testid="stSidebar"] h2 {
    color: #d97706;
    font-weight: 800;
}

/* Result cards */
.result-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    transition: transform 0.2s ease-in-out;
}
.result-card:hover {
    transform: translateY(-5px);
}
.result-card .icon {
    font-size: 2rem;
}
.result-card .number {
    font-size: 2.5rem;
    font-weight: 900;
    background: linear-gradient(90deg, #d97706, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 0.3rem;
}

/* Webcam frame */
.output-image img {
    border-radius: 16px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.15);
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    max-height: 500px;
    object-fit: contain;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="header">
    <h1>🚦 MobilityVision</h1>
    <p>Your AI-powered lens on urban mobility</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Controls")
option = st.sidebar.radio("Select Input Method", ("Upload Image", "Use Webcam"))

# -----------------------------
# Upload Image Flow
# -----------------------------
if option == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results_people = people_model(img, conf=0.30)
        results_vehicles = vehicles_model(img, conf=0.25)

        annotated_people = results_people[0].plot()
        annotated_vehicles = results_vehicles[0].plot()
        combined = cv2.addWeighted(annotated_people, 0.5, annotated_vehicles, 0.5, 0)

        col_center = st.columns([1,2,1])
        with col_center[1]:
          st.image(combined, channels="BGR", width=500)

       

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="icon">👤</div>
                People Detected
                <div class="number">{len(results_people[0].boxes)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="icon">🚗</div>
                Vehicles Detected
                <div class="number">{len(results_vehicles[0].boxes)}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📂 Please upload an image to start detection.")

# -----------------------------
# Webcam Flow
# -----------------------------
elif option == "Use Webcam":
    st.sidebar.subheader("Webcam Controls")
    if st.sidebar.button("▶️ Start Webcam") and not st.session_state.webcam_running:
        st.session_state.webcam_running = True
    if st.sidebar.button("⏹ Stop Webcam") and st.session_state.webcam_running:
        st.session_state.webcam_running = False
    if st.sidebar.button("🔄 Reset Counts"):
        st.session_state.unique_people_ids.clear()
        st.session_state.unique_vehicle_ids.clear()

    frame_slot = st.empty()
    col1, col2 = st.columns(2)
    people_slot = col1.empty()
    vehicle_slot = col2.empty()

    if st.session_state.webcam_running:
        tracker_yaml_path = write_tracker_yaml()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error("Error: Unable to open webcam")
        else:
            conf, iou, imgsz = 0.25, 0.5, 640

            while cap.isOpened() and st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    break

                results_people = people_model.track(
                    frame, conf=conf, iou=iou, imgsz=imgsz, persist=True, tracker=tracker_yaml_path, verbose=False
                )
                results_vehicles = vehicles_model.track(
                    frame, conf=conf, iou=iou, imgsz=imgsz, persist=True, tracker=tracker_yaml_path, verbose=False
                )

                annotated_people = results_people[0].plot()
                annotated_vehicles = results_vehicles[0].plot()
                combined = cv2.addWeighted(annotated_people, 0.5, annotated_vehicles, 0.5, 0)

                p_ids = extract_ids(results_people[0].boxes)
                v_ids = extract_ids(results_vehicles[0].boxes)
                for pid in p_ids:
                    st.session_state.unique_people_ids.add(pid)
                for vid in v_ids:
                    st.session_state.unique_vehicle_ids.add(vid)

                col_center = st.columns([1,2,1])
                with col_center[1]:
                  frame_slot.image(combined, channels="BGR", width=500)

                people_slot.markdown(f"""
                <div class="result-card">
                    <div class="icon">👤</div>
                    Unique People
                    <div class="number">{len(st.session_state.unique_people_ids)}</div>
                </div>
                """, unsafe_allow_html=True)
                vehicle_slot.markdown(f"""
                <div class="result-card">
                    <div class="icon">🚗</div>
                    Unique Vehicles
                    <div class="number">{len(st.session_state.unique_vehicle_ids)}</div>
                </div>
                """, unsafe_allow_html=True)

            cap.release()
    else:
        col1, col2 = st.columns(2)
        col1.markdown(f"""
        <div class="result-card">
            <div class="icon">👤</div>
            Unique People
            <div class="number">{len(st.session_state.unique_people_ids)}</div>
        </div>
        """, unsafe_allow_html=True)
        col2.markdown(f"""
        <div class="result-card">
            <div class="icon">🚗</div>
            Unique Vehicles
            <div class="number">{len(st.session_state.unique_vehicle_ids)}</div>
        </div>
        """, unsafe_allow_html=True)
