# 🎯 AI/ML Face Detection Web Application

A production-ready face detection web application built with **Streamlit** and **OpenCV**, supporting both image upload and live webcam detection modes.

---

## 📋 Project Overview

This application uses the classic **Viola-Jones Haar Cascade** algorithm implemented via OpenCV to detect human faces in still images and live webcam streams. It features a polished dark-mode UI, configurable detection parameters, per-face confidence display, and a session detection history table.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Image Upload Detection** | Upload JPG, JPEG, or PNG images and detect faces instantly |
| **Live Webcam Detection** | Real-time face detection via browser camera (streamlit-webrtc) |
| **Configurable Parameters** | Tune Scale Factor, Min Neighbors, and Min Face Size from the sidebar |
| **Confidence Display** | Per-face confidence score derived from the cascade reject-level weights |
| **Bounding Box Labels** | Face index + confidence shown above each detected face |
| **Detection Statistics** | Face count, image resolution, detection time, filename |
| **Download Button** | Download the annotated image as PNG |
| **Detection History** | Session table of all previous detections |
| **Dark Mode UI** | Custom CSS dark theme with gradient accents |
| **Modular Code Structure** | Separate `app.py` (UI) and `face_detector.py` (core logic) |

---

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)** — Interactive Python web app framework
- **[OpenCV (opencv-python-headless)](https://opencv.org/)** — Computer vision library with Haar Cascade face detection
- **[NumPy](https://numpy.org/)** — Numerical array operations
- **[Pillow](https://python-pillow.org/)** — Image loading and format handling
- **[streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)** — WebRTC-based live video streaming
- **[PyAV (av)](https://pyav.org/)** — Video frame decoding for WebRTC

---

## 📁 Project Structure

```
streamlit-app/
├── app.py                   # Main Streamlit application (UI + pages)
├── face_detector.py         # Core face detection logic (OpenCV)
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── .streamlit/
    └── config.toml          # Streamlit server + theme configuration
```

> **Note:** The Haar Cascade XML (`haarcascade_frontalface_default.xml`) is loaded directly from OpenCV's bundled data directory (`cv2.data.haarcascades`) — no external file download is required.

---

## 🚀 Installation Steps

### Prerequisites

- Python 3.9 or later
- pip

### 1. Clone / copy the project

```bash
git clone <your-repo-url>
cd streamlit-app
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The app will open at `http://localhost:5000` by default (configured in `.streamlit/config.toml`).

---

## 📖 Usage Instructions

### Image Detection

1. Open the app and navigate to **📸 Image Detection** in the sidebar.
2. Click **Browse files** to upload a JPG, JPEG, or PNG image.
3. The app will automatically detect faces and display:
   - The original image (left)
   - The annotated image with green bounding boxes (right)
   - Detection statistics (face count, resolution, time)
   - A per-face detail table with position, size, and confidence
4. Click **⬇️ Download Processed Image** to save the result.

### Webcam Detection

1. Navigate to **📷 Webcam Detection** in the sidebar.
2. Click **START** to launch the WebRTC camera stream.
3. Allow browser camera access when prompted.
4. Faces will be highlighted in real time with green bounding boxes.
5. Click **STOP** to end the stream.

### Tuning Detection

Use the sidebar sliders to adjust sensitivity:

| Parameter | Lower value | Higher value |
|---|---|---|
| **Scale Factor** | More thorough, slower | Faster, may miss small faces |
| **Min Neighbors** | More detections (more noise) | Fewer false positives |
| **Min Face Size** | Detects smaller faces | Ignores small regions |

---

## 📸 Screenshots

> _Add screenshots here after deployment_

| Home Page | Image Detection | Webcam Detection |
|---|---|---|
| _(screenshot)_ | _(screenshot)_ | _(screenshot)_ |

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push the `streamlit-app/` folder contents to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app** and connect your GitHub repo.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**.

> Streamlit Community Cloud installs `requirements.txt` automatically — no additional configuration needed.

---

## ⚙️ How Face Detection Works

1. **Grayscale Conversion** — Haar cascades operate on single-channel images.
2. **Histogram Equalisation** — Normalises brightness across varied lighting.
3. **Image Pyramid** — The image is repeatedly scaled down by `scale_factor` to detect faces at different sizes.
4. **Sliding Window** — At each scale, a fixed-size window scans across the image.
5. **Cascade Evaluation** — Each window passes through a series of classifier stages; it is immediately rejected if any stage fails.
6. **Result Grouping** — Overlapping detections are merged using `min_neighbors`; only sufficiently supported rectangles survive.

---

## 📄 License

MIT — free to use, modify, and distribute.
