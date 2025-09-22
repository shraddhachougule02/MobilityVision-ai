# 🚦 MobilityVision  

MobilityVision is an **AI-powered real-time people and vehicle detection system** built using **YOLOv8**. It enables the recognition, tracking, and counting of people and vehicles through uploaded images or live webcam streams, making it highly useful for **urban surveillance, mobility management, and smart city applications**.  

---

## 📌 Features
- **Trained Custom Models**  
  - Separate YOLOv8 models trained for *people* (`best2.pt`) and *vehicles* (`best.pt`).  
  - Fine-tuned on relevant datasets for improved accuracy in real-world environments.  

- **Multiple Input Modes**  
  - **Upload Image Mode**: Upload an image for detection and view annotated results.  
  - **Webcam Mode**: Real-time people and vehicle tracking using live video feed.  

- **Object Tracking with ByteTrack**  
  - Persistent tracking of unique IDs across frames.  
  - Counts unique people and vehicles, avoiding duplicates.  

- **Streamlit Web Interface**  
  - Modern, user-friendly frontend with styled dashboard.  
  - Displays live counts, annotations, and results in real-time.  

- **Overlay Annotations**  
  - People and vehicle results overlaid on the same frame.  
  - Highlighted bounding boxes with detection confidence levels.  

---

## 🎯 Use Cases
- **Smart Cities**: Monitor traffic and pedestrian density in real time.  
- **Traffic Management**: Provide insights on road congestion and mobility patterns.  
- **Event Security**: Count and track crowds to prevent over-crowding.  
- **Urban Planning**: Gather data for policy makers on vehicle-to-people ratios.  
- **Surveillance**: Assist law enforcement with intelligent monitoring of public spaces.  

---

## 🌍 Impact on Society
MobilityVision has the potential to:  
- Reduce traffic jams by providing **real-time traffic insights**.  
- Enhance public safety by **tracking pedestrian flow** in crowded areas.  
- Support **sustainable urban mobility initiatives** by analyzing daily traffic volumes.  
- Provide **accurate crowd estimates** during public events and ensure safety measures.  

This project is aligned with the vision of **smarter, safer, and more efficient cities**.  

---

## ⚙️ Tech Stack
- **Backend**: Python, FastAPI, OpenCV  
- **Frontend**: Streamlit (custom CSS styling)  
- **Models**: YOLOv8 (Ultralytics) for People and Vehicle detection  
- **Tracking**: ByteTrack algorithm for persistent tracking  
- **Visualization**: Real-time annotated image overlays  

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```
git clone https://github.com/your-username/MobilityVision.git
cd MobilityVision
```

### 2. Create Virtual Environment
```
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

*(Dependencies include ultralytics, opencv-python, streamlit, numpy, torch, etc.)*

### 4. Add Trained Models
Place your trained models in the project directory:
- `best2.pt` (people detection model)  
- `best.pt` (vehicle detection model)  

---

## ▶️ Usage Instructions

### 1. Run Streamlit Application
```
streamlit run app.py
```
- Open browser at [**http://localhost:8501**](http://localhost:8501)  
- Choose between **Upload Image** or **Use Webcam** modes from the sidebar  

---

### 2. Run Direct Webcam Detection (Optional, without Streamlit UI)
```
python webcam_tracking.py
```

- Opens real-time tracking window.
- Press **q** to exit.  

---

## 📊 Results and Highlights
- Achieved **real-time performance** (>20 FPS) for webcam streams.  
- Accurate detection of both people and vehicles, even in **low-light or busy environments**.  
- Custom models trained on diverse datasets for **higher generalization**.  
- Facilitates **live streaming** analysis for urban mobility monitoring.  

---

## 📷 Screenshots & 🎥 Demo Videos
*(To be uploaded)*  
- Example image detections  
- Live webcam streaming demo  
- Dashboard with real-time mobility counts  

---

## 📌 Future Improvements
- Integration with **cloud database** for storing traffic analytics.  
- Deployment with **Flask/FastAPI APIs** for scalable backend.  
- Adding detection for **more mobility types** (bicycles, buses, trucks).  
- Real-time alerts and notifications for crowd/traffic anomalies.  

---

## 🧑‍💻 Author
**Developed by:** [Your Name]  
- Data Science & AI Enthusiast  
- Focused on **Computer Vision, NLP, and Smart City AI Applications**  
- Connect with me on [LinkedIn](https://www.linkedin.com/) / [GitHub](https://github.com/)  

---

## 🏷️ Project Identity
- **Project Name**: MobilityVision  
- **Domain**: AI / Computer Vision / Smart Cities  
- **Category**: Real-time Object Tracking & Counting  

---
```

Would you like me to also create a **`requirements.txt` file** for you with all the necessary dependencies so that anyone can install and run your project without missing packages?
