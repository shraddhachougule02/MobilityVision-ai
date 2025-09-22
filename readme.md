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
- **Smart Parking**: Assistance in parking for smart parking management.
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


## 📊 Results and Highlights
- Achieved **real-time performance** (>20 FPS) for webcam streams.  
- Accurate detection of both people and vehicles, even in **low-light or busy environments**.  
- Custom models trained on diverse datasets for **higher generalization**.  
- Facilitates **live streaming** analysis for urban mobility monitoring.  

---

## 📷 Screenshots &  Demo 

[![Watch the demo](demo.gif)](detection.mp4)

 <img width="1897" height="846" alt="image" src="https://github.com/user-attachments/assets/b12e49a5-6c94-4d8d-a50c-c582d49c3e04" />

- Example image detections
 
<img width="1887" height="764" alt="image" src="https://github.com/user-attachments/assets/2415c7c6-bf45-4936-8dae-b22a9c9eb3e2" />

  
- Live webcam streaming demo
  
<img width="1873" height="765" alt="image" src="https://github.com/user-attachments/assets/4fb2f296-9858-48f8-a6ae-39afc65ed34a" />

---

## 🧑‍💻 Author
**Developed by:** Shraddha Sanjay Chougule 
- Data Science & AI Enthusiast  
- Focused on **Computer Vision, NLP, and Smart City AI Applications**  
- Connect with me on [LinkedIn][(https://www.linkedin.com/)](https://www.linkedin.com/in/cshraddha) / [GitHub][(https://github.com/)  
](https://github.com/shraddhachougule02)
---

## 🏷️ Project Identity
- **Project Name**: MobilityVision  
- **Domain**: AI / Computer Vision / Smart Cities  
- **Category**: Real-time Object Tracking & Counting  

---