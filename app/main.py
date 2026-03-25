import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model — 'yolov8n.pt' is the smallest/fastest version
# 'n' stands for nano. First run will auto-download the model weights (~6MB)
model = YOLO("yolov8n.pt")

# VideoCapture(0) opens your default webcam — 0 means first camera found/ substitue with video file path to read from a video instead
cap = cv2.VideoCapture("traffic_video.mp4")

# Loop continuously to read frames one at a time (video = many images played fast)
while True:
    # cap.read() returns two things:
    # ret — True/False whether the frame was read successfully
    # frame — the actual image data as a NumPy array (grid of pixel values)
    ret, frame = cap.read()

    # If frame wasn't read successfully, stop
    if not ret:
        break

    # Pass the frame to YOLO — it returns a list of results (one per frame)
    results = model(frame, verbose=False)

    # annotated_frame draws the bounding boxes and labels directly onto the frame
    # annotated_frame = results[0].plot()

    # results[0].boxes contains all detections for this frame
    # each box has: xyxy (coordinates), conf (confidence score), cls (class id)
    for box in results[0].boxes:
        # xyxy gives us the top-left and bottom-right corners of the bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # conf is the confidence score (0 to 1) — how sure the model is
        conf = float(box.conf[0])

        # cls is a number — we convert it to a label name like "car" or "truck"
        label = model.names[int(box.cls[0])]

        # Draw the rectangle on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put the label and confidence score above the box
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame in a window called "Traffic Detection"
    cv2.imshow("Traffic Detection", frame)

    # Wait 1ms between frames — if user presses 'q', exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows cleanly
cap.release()
cv2.destroyAllWindows()