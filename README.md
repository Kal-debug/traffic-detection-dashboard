# 🚦 Traffic Detection Dashboard

A real-time object detection application that analyzes traffic videos using computer vision and machine learning.

## 🔍 What It Does
- Processes uploaded traffic videos frame by frame
- Detects and classifies objects (cars, buses, trucks, people, motorcycles, traffic lights) using YOLOv8
- Displays live annotated video with bounding boxes and confidence scores
- Shows detection counts as metric cards and a bar chart

## 🛠️ Tech Stack
- **Python** — core language
- **OpenCV** — video processing and frame manipulation
- **YOLOv8 (Ultralytics)** — pre-trained object detection model
- **Streamlit** — browser-based dashboard UI
- **Pandas** — data formatting for charts

## 🚀 How To Run

1. Clone the repository
2. Create and activate a virtual environment
```
   python3 -m venv venv
   source venv/bin/activate
```
3. Install dependencies
```
   pip install opencv-python ultralytics streamlit pandas
```
4. Run the dashboard
```
   streamlit run app/dashboard.py
```
5. Upload any traffic video (MP4) and watch it go

## 📸 Demo
![Dashboard Screenshot](demo.png)