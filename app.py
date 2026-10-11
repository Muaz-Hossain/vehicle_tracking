import tempfile                                              

import cv2
import streamlit as st                                       
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

st.title("🚗 Vehicle Tracking & Counting")                   
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# files upload
uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])  

if uploaded and st.button("Start"):                          # new
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")   # new
    tfile.write(uploaded.read())                             # new
    video_path = tfile.name                                  # new

    cap = cv2.VideoCapture(video_path)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    scale = 640 / width if width > 640 else 1.0
    width, height = int(width * scale), int(height * scale)
    
    tracker = DeepSort(max_age=50, n_init=3, nms_max_overlap=0.5)

    line_y = height // 2
    frame_skip = 3
    frame_count = 0
    cross_ids = set()
    prev_y = {}                                              # update
    vehicles_classes = [2, 3, 5, 7]

    frame_box = st.empty()                                   

    while frame_count < 900::
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            frame = cv2.resize(frame, (width, height))
            
            continue

        # run yolo
    
        results = model.predict(frame, verbose=False, classes=vehicles_classes)
        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

        # update tracker
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cy = (y1 + y2) // 2

            # draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),     
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            
            if track_id in prev_y and track_id not in cross_ids:
                if (prev_y[track_id] < line_y <= cy) or (prev_y[track_id] > line_y >= cy):
                    cross_ids.add(track_id)
            prev_y[track_id] = cy

        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"Total: {len(cross_ids)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_box.image(frame, channels="BGR")               # update

    cap.release()
    st.success(f"Total Vehicle count: {len(cross_ids)}")     
