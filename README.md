# 🚗 Vehicle Tracking & Counting with YOLOv8 + DeepSORT

A real‑time vehicle tracking and counting system that detects cars, buses, trucks, and motorcycles, assigns unique IDs using DeepSORT, and counts vehicles crossing a virtual line

---

## ✨ Features

- **Vehicle detection** using YOLOv8 (nano model)
- **Tracking** with DeepSORT – each vehicle gets a persistent ID
- **Line‑based counting** – increment counter when vehicle center crosses a virtual line
- **Filter by class** – counts only cars, motorcycles, buses, trucks (ignores persons, bicycles, etc.)
- **Output video** with bounding boxes, IDs, and counter overlay
- **Console logging** – shows each crossing event and total count

---

## 🛠️ Tech Stack

| Component       | Technology                          |
|----------------|-------------------------------------|
| Detection       | YOLOv8 (Ultralytics)                |
| Tracking        | DeepSORT (`deep-sort-realtime`)     |
| Image Processing| OpenCV                              |
| Language        | Python 3.11                         |
| Environment     | Google Colab / Local (GPU optional) |

---

## 📁 Repository Structure



## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Muaz-Hossain/vehicle-tracking.git
cd vehicle-tracking
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

requirements.txt should contain:

```
ultralytics
opencv-python
deep-sort-realtime
numpy
```

3. Run the notebook

· Open tracking.ipynb in Google Colab or Jupyter.
· Upload your video file (MP4).
· Run all cells.

The output video will be saved as tracked_output.mp4 and the console will print total count.

4. (Optional) Run as Python script

```bash
python tracking.py --video your_video.mp4 --line_y 360 --conf 0.5
```

---

📊 Sample Output

```
Vehicle 1 crossed. Total: 1
Vehicle 2 crossed. Total: 2
...
Total vehicles counted: 437
```

sample_frame.jpg

---

🧠 How It Works

1. Read video frame by frame.
2. YOLOv8 detects vehicles (classes 2,3,5,7).
3. DeepSORT tracker assigns an ID to each detected vehicle and maintains it across frames.
4. A virtual line (horizontal line at line_y = height//2) is drawn.
5. When the center of a vehicle's bounding box crosses this line, the counter increments (once per ID).
6. The annotated frame is written to an output video.

---

🎯 Customisation

· Change counting line position – modify line_y (0 to frame height).
· Add direction counting – store previous centroid to detect upward/downward crossing.
· Filter by specific vehicle type – adjust vehicle_classes list.
· Adjust tracking sensitivity – modify max_age, n_init parameters in DeepSORT initialisation.

---

📝 TODO / Future Improvements

· Add direction‑aware counting (incoming)
· Export counting results to CSV
· Build Streamlit UI for interactive line placement
· Integrate with RTSP camera streams

---

🙏 Acknowledgements

· Ultralytics YOLOv8
· Deep SORT (real‑time implementation)
· OpenCV community

