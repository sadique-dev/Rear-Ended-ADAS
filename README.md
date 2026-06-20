# 🚗 Camera-Based Rear-End ADAS Collision Warning System

A modular **Computer Vision-based Rear-End Advanced Driver Assistance System (ADAS)** developed using **YOLOv8**, **ByteTrack**, and **Monocular Vision** to detect rear vehicles, estimate distance, calculate Time-To-Collision (TTC), and provide collision risk warnings from a single rear-facing camera.

This project was developed as a B.Tech Final Year Project and demonstrates a complete AI-powered rear-end collision warning pipeline.

---

# 📌 Features

- 🚗 Vehicle Detection using YOLOv8
- 🔄 Multi-Object Tracking using ByteTrack
- 🎯 Lead Vehicle Selection using Region of Interest (ROI)
- 📏 Monocular Distance Estimation
- 📉 Relative Speed Estimation
- ⏱️ Time-To-Collision (TTC) Estimation
- ⚠️ Collision Risk Classification
- 🎨 Real-Time Video Overlay
- 💾 Processed Video Output Generation
- 🧩 Modular and Scalable Project Architecture

---

# 🏗️ System Pipeline

Input Video

↓

Video Reader

↓

YOLOv8 Vehicle Detection

↓

ByteTrack Multi-Object Tracking

↓

Lead Vehicle Selection

↓

Distance Estimation

↓

Relative Speed Estimation

↓

Time-To-Collision (TTC)

↓

Risk Classification

↓

Visualization Overlay

↓

Output Video

---

# 📂 Project Structure

```
Rear-End-ADAS-System/

├── config/
├── data/
│   ├── samples/
│   └── outputs/
│
├── models/
│
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── estimation/
│   ├── risk/
│   ├── visualization/
│   ├── pipeline/
│   ├── io/
│   └── utils/
│
├── tests/
├── examples/
├── README.md
└── requirements.txt
```

---

# 🛠️ Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- ByteTrack
- NumPy
- PyTorch
- YAML Configuration
- Object-Oriented Programming
- Computer Vision

---

# ⚙️ How It Works

### 1. Vehicle Detection

YOLOv8 detects rear vehicles in every frame.

Supported classes:

- Car
- Motorcycle
- Bus
- Truck

---

### 2. Multi-Object Tracking

ByteTrack assigns a persistent Track ID to each detected vehicle, allowing the same vehicle to be tracked across consecutive frames.

---

### 3. Lead Vehicle Selection

A configurable Region of Interest (ROI) is used to identify the primary vehicle directly behind the ego vehicle.

If multiple vehicles are inside the ROI, the closest vehicle is selected using bounding-box size.

---

### 4. Distance Estimation

Distance is estimated using the pinhole camera model:

Distance = (Real Vehicle Width × Focal Length) / Bounding Box Width

Vehicle-specific widths are used for improved estimation.

---

### 5. Relative Speed Estimation

Relative speed is calculated using changes in estimated distance over time.

Relative Speed = ΔDistance / ΔTime

---

### 6. Time-To-Collision (TTC)

When the lead vehicle is approaching:

TTC = Distance / |Relative Speed|

Otherwise:

TTC = INF

---

### 7. Risk Classification

The estimated TTC is classified into:

| Risk | TTC |
|------|------|
| 🟢 SAFE | > 5 seconds |
| 🟡 CAUTION | 2–5 seconds |
| 🔴 DANGER | ≤ 2 seconds |

---

# ▶️ Running the Project

Install dependencies

```bash
pip install -r requirements.txt
```

Run the complete pipeline

```bash
python examples/final_demo.py --input data/samples/drive.mp4 --output data/outputs/final_demo.mp4
```

or

```bash
python -m src.main --input data/samples/drive.mp4 --output data/outputs/final_demo.mp4
```

---

# 📊 Example Output

The generated output video displays:

- Vehicle Detection
- Track ID
- Vehicle Class
- Detection Confidence
- Estimated Distance
- Relative Speed
- Time-To-Collision (TTC)
- Collision Risk Level

---

# ⚠️ Limitations

- Uses a single monocular camera.
- Distance estimation is approximate and depends on assumed vehicle widths.
- Performance may decrease under poor lighting or severe weather.
- Not intended for real-world safety-critical driving applications.
- Designed for educational and research purposes.

---

# 🚀 Future Improvements

- Lane Detection
- Camera Calibration
- Stereo Vision Support
- Radar/LiDAR Sensor Fusion
- DeepSORT / OC-SORT Comparison
- Driver Alert Sound System
- Real-Time Webcam Support
- Model Optimization for Edge Devices

---

# 👨‍💻 Author

**Mohd Sadique**

B.Tech Computer Science & Engineering

AI | Machine Learning | Computer Vision

---

# ⭐ Acknowledgements

- Ultralytics YOLOv8
- ByteTrack
- OpenCV
- PyTorch
