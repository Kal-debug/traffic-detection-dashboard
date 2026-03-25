import cv2
import tempfile # creates temporary files on disk — needed because OpenCV requires a file path, not raw bytes
import streamlit as st
from ultralytics import YOLO
from collections import defaultdict # like a dict but auto-sets missing keys to 0, avoids KeyError when counting
import pandas as pd

st.title("🚦 Traffic Detection Dashboard")
st.write("Upload a traffic video and detect objects in real time.")

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:

    # Streamlit gives us the file as bytes, not a path
    # OpenCV needs a file path, so we save it to a temporary file first
    # tempfile.NamedTemporaryFile creates a temp file and gives us its path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)

    # st.empty() creates a placeholder in the UI we can update each frame
    frame_placeholder = st.empty()

    # st.empty() for the stats section below the video
    stats_placeholder = st.empty()

    # defaultdict(int) is like a regular dict but auto-sets missing keys to 0
    # we use it to count detections per object class across all frames
    total_counts = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = model.names[int(box.cls[0])]

            # Count each detected object by label
            total_counts[label] += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame from BGR (OpenCV format) to RGB (what browsers expect)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update the placeholder with the new frame
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # st.columns splits the UI into side by side columns
        # we create one column per detected object class
        cols = stats_placeholder.columns(len(total_counts))

        # zip pairs each column with its label and count
        for col, (label, count) in zip(cols, total_counts.items()):
            # st.metric displays a clean labeled number card
            col.metric(label=label, value=count)

        # Convert counts to a DataFrame — the format st.bar_chart expects
    # pd.DataFrame creates a table from a dictionary
    df = pd.DataFrame.from_dict(total_counts, orient="index", columns=["Total Detections"])

    stats_placeholder_chart = st.empty()
    stats_placeholder_chart.bar_chart(df)

    cap.release()
    st.success("✅ Video processing complete!")