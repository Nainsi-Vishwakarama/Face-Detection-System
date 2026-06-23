"""
app.py
------
Main Streamlit application for AI/ML Face Detection.
Provides image upload detection and live webcam detection via streamlit-webrtc.
"""

import io
import time
import datetime
from typing import Optional

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Local module for face detection logic
from face_detector import detect_faces, draw_labeled_boxes, load_cascade

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Face Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — Dark Mode & Polished UI
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global dark background ── */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 18px;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #58a6ff;
        border-bottom: 2px solid #58a6ff33;
        padding-bottom: 6px;
        margin-bottom: 18px;
    }

    /* ── Info / stat cards ── */
    .stat-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    .stat-label {
        font-size: 0.78rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #58a6ff;
    }

    /* ── Success / warning badges ── */
    .badge-success {
        display: inline-block;
        background: #1f4d2e;
        color: #56d364;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .badge-warning {
        display: inline-block;
        background: #4d3800;
        color: #e3b341;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 28px;
        text-align: center;
    }

    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #79c0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .hero p {
        color: #8b949e;
        font-size: 1.05rem;
        margin: 0;
    }

    /* ── Tables ── */
    .history-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }

    .history-table th {
        background: #161b22;
        color: #8b949e;
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid #30363d;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
    }

    .history-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #21262d;
        color: #c9d1d9;
    }

    .history-table tr:hover td {
        background: #161b22;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.88;
    }

    /* ── File uploader ── */
    .stFileUploader {
        border: 2px dashed #30363d;
        border-radius: 10px;
        padding: 10px;
    }

    /* ── Divider ── */
    hr {
        border-color: #30363d;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #161b22 !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []  # list of result dicts


# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Face Detection AI")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=["🏠 Home", "📸 Image Detection", "📷 Webcam Detection", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("### ⚙️ Detection Settings")
    scale_factor = st.slider(
        "Scale Factor",
        min_value=1.05,
        max_value=1.5,
        value=1.1,
        step=0.05,
        help="How much the image is scaled down at each detection pass. "
             "Lower = more sensitive but slower.",
    )
    min_neighbors = st.slider(
        "Min Neighbors",
        min_value=1,
        max_value=10,
        value=5,
        help="How many overlapping detections are required to confirm a face. "
             "Higher = fewer false positives.",
    )
    min_face_size = st.slider(
        "Min Face Size (px)",
        min_value=10,
        max_value=100,
        value=30,
        help="Smallest face dimension to detect.",
    )
    show_labels = st.checkbox("Show confidence labels", value=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>Powered by OpenCV Haar Cascade</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Helper: Add to History
# ─────────────────────────────────────────────
def add_to_history(filename: str, count: int, resolution: str, status: str):
    """Append a detection result to the session history list."""
    st.session_state.detection_history.append(
        {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "file": filename,
            "faces": count,
            "resolution": resolution,
            "status": status,
        }
    )
    # Keep only the last 20 entries
    st.session_state.detection_history = st.session_state.detection_history[-20:]


# ─────────────────────────────────────────────
# Helper: Stat Card HTML
# ─────────────────────────────────────────────
def stat_card(label: str, value: str) -> str:
    return f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
    </div>
    """


# ─────────────────────────────────────────────
# PAGE: Home
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <h1>🎯 AI Face Detection</h1>
            <p>Real-time face detection powered by OpenCV Haar Cascade Classifier</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📸 Image Detection**\nUpload a JPG, JPEG, or PNG image to detect faces instantly.")
    with col2:
        st.info("**📷 Webcam Detection**\nUse your live camera for real-time face detection in the browser.")
    with col3:
        st.info("**📊 Detection History**\nAll detections are tracked in a session history table.")

    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    st.markdown(
        "1. Use the **sidebar** to navigate between sections.\n"
        "2. Adjust the **detection settings** to tune sensitivity.\n"
        "3. Go to **Image Detection** to upload an image, or\n"
        "4. Go to **Webcam Detection** to start live camera analysis."
    )

    st.markdown("---")
    st.markdown("### 📈 Session Summary")
    total = len(st.session_state.detection_history)
    total_faces = sum(r["faces"] for r in st.session_state.detection_history)
    m1, m2 = st.columns(2)
    m1.metric("Images Analysed", total)
    m2.metric("Total Faces Found", total_faces)


# ─────────────────────────────────────────────
# PAGE: Image Detection
# ─────────────────────────────────────────────
elif page == "📸 Image Detection":
    st.markdown('<div class="section-header">📸 Image Face Detection</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Maximum recommended size: 10 MB",
    )

    if uploaded_file is not None:
        # ── Validate file is a real image ──
        try:
            pil_image = Image.open(uploaded_file)
            pil_image.verify()          # Raises if not a valid image
            uploaded_file.seek(0)       # Reset after verify()
            pil_image = Image.open(uploaded_file)
        except Exception:
            st.error("❌ Invalid image file. Please upload a valid JPG, JPEG, or PNG.")
            st.stop()

        col_orig, col_proc = st.columns(2, gap="medium")

        with col_orig:
            st.markdown("**Original Image**")
            st.image(pil_image, use_container_width=True)

        with st.spinner("🔍 Detecting faces…"):
            start_time = time.time()
            try:
                result = detect_faces(
                    pil_image,
                    scale_factor=scale_factor,
                    min_neighbors=min_neighbors,
                    min_size=(min_face_size, min_face_size),
                )
                elapsed = time.time() - start_time
            except Exception as e:
                st.error(f"❌ Detection error: {e}")
                st.stop()

        # Draw labeled boxes if the option is enabled
        if show_labels and result["count"] > 0:
            annotated_image = draw_labeled_boxes(
                pil_image, result["faces"], result["confidence_scores"], show_labels=True
            )
        else:
            annotated_image = result["processed_image"]

        with col_proc:
            st.markdown("**Processed Image**")
            st.image(annotated_image, use_container_width=True)

        # ── Statistics ──
        st.markdown("---")
        st.markdown("### 📊 Detection Statistics")

        w, h = result["image_size"]
        resolution_str = f"{w} × {h}"
        status = "Faces Detected" if result["count"] > 0 else "No Faces Found"
        status_badge = (
            '<span class="badge-success">✅ Faces Detected</span>'
            if result["count"] > 0
            else '<span class="badge-warning">⚠️ No Faces Found</span>'
        )

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Faces Detected", result["count"])
        s2.metric("Image Resolution", resolution_str)
        s3.metric("Detection Time", f"{elapsed:.2f}s")
        s4.metric("File Name", uploaded_file.name[:18] + ("…" if len(uploaded_file.name) > 18 else ""))

        st.markdown(f"**Status:** {status_badge}", unsafe_allow_html=True)

        # ── Per-face confidence table ──
        if result["count"] > 0 and show_labels:
            st.markdown("---")
            st.markdown("### 🔎 Face-by-Face Details")
            rows = ""
            for i, (face, score) in enumerate(
                zip(result["faces"], result["confidence_scores"]), start=1
            ):
                x, y, w_f, h_f = face
                rows += (
                    f"<tr>"
                    f"<td>Face {i}</td>"
                    f"<td>({x}, {y})</td>"
                    f"<td>{w_f} × {h_f} px</td>"
                    f"<td>{score}%</td>"
                    f"</tr>"
                )
            st.markdown(
                f"""
                <table class="history-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Position (x, y)</th>
                            <th>Size</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )

        # ── Download button ──
        st.markdown("---")
        buf = io.BytesIO()
        annotated_image.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Processed Image",
            data=buf.getvalue(),
            file_name=f"detected_{uploaded_file.name.rsplit('.', 1)[0]}.png",
            mime="image/png",
        )

        # ── Save to history ──
        add_to_history(
            filename=uploaded_file.name,
            count=result["count"],
            resolution=resolution_str,
            status=status,
        )

    else:
        st.info("👆 Upload an image above to begin face detection.")

    # ── Detection History ──
    if st.session_state.detection_history:
        st.markdown("---")
        st.markdown("### 🗂️ Detection History")
        rows = ""
        for entry in reversed(st.session_state.detection_history):
            face_badge = (
                f'<span class="badge-success">{entry["faces"]} face(s)</span>'
                if entry["faces"] > 0
                else f'<span class="badge-warning">0 faces</span>'
            )
            rows += (
                f"<tr>"
                f"<td>{entry['timestamp']}</td>"
                f"<td>{entry['file']}</td>"
                f"<td>{face_badge}</td>"
                f"<td>{entry['resolution']}</td>"
                f"<td>{entry['status']}</td>"
                f"</tr>"
            )
        st.markdown(
            f"""
            <table class="history-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>File</th>
                        <th>Faces</th>
                        <th>Resolution</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Clear History"):
            st.session_state.detection_history = []
            st.rerun()


# ─────────────────────────────────────────────
# PAGE: Webcam Detection
# ─────────────────────────────────────────────
elif page == "📷 Webcam Detection":
    st.markdown('<div class="section-header">📷 Live Webcam Face Detection</div>', unsafe_allow_html=True)

    st.info(
        "Click **START** below to enable your camera. "
        "Faces will be highlighted with green bounding boxes in real time. "
        "Your browser may ask for camera permission — please allow it."
    )

    # ── Load cascade once, outside the per-frame callback ──
    try:
        _cascade = load_cascade()
    except RuntimeError as e:
        st.error(f"❌ Could not load face detector: {e}")
        st.stop()

    # Counter displayed alongside the stream
    face_count_placeholder = st.empty()

    # Shared mutable container so the callback can report back to the main thread
    _state = {"count": 0}

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        """
        Per-frame callback for streamlit-webrtc.
        Converts the incoming frame, detects faces, draws boxes, returns annotated frame.
        """
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = _cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_face_size, min_face_size),
        )

        _state["count"] = len(faces) if len(faces) > 0 else 0

        # Draw green rectangles around each detected face
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Face  {int(w)}px"
            cv2.putText(
                img, label, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA
            )

        # Overlay face count in top-left corner of the video
        cv2.putText(
            img,
            f"Faces: {_state['count']}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # WebRTC configuration with public STUN servers for NAT traversal
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="face-detection-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Show live face count below the stream while it is active
    if ctx.state.playing:
        face_count_placeholder.metric("🟢 Faces in Frame", _state["count"])


# ─────────────────────────────────────────────
# PAGE: About
# ─────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown('<div class="section-header">ℹ️ About This Project</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ## AI/ML Face Detection Web Application

        This application demonstrates computer-vision-based face detection directly in the browser
        using a classic machine-learning approach — **Viola-Jones Haar Cascade** — implemented through
        **OpenCV**.

        ---
        ### 🧠 How It Works

        1. **Preprocessing** — The input image is converted to grayscale and histogram-equalised
           to improve detection under varied lighting conditions.
        2. **Sliding Window** — The Haar cascade classifier scans the image at multiple scales
           using a sliding window approach.
        3. **Feature Evaluation** — At each window position, a set of Haar-like rectangular
           features is evaluated rapidly using an integral image.
        4. **Cascade of Classifiers** — A series of increasingly complex classifiers (stages) is
           applied; a window is rejected as soon as one stage fails, making detection very fast.
        5. **Non-Maximum Suppression** — Overlapping detection windows are merged using the
           `minNeighbors` threshold to produce final bounding boxes.

        ---
        ### 🛠️ Technologies Used

        | Component | Library / Tool |
        |---|---|
        | Web framework | Streamlit |
        | Face detection | OpenCV (`cv2`) Haar Cascade |
        | Image processing | Pillow, NumPy |
        | Live video streaming | streamlit-webrtc, PyAV (`av`) |
        | Language | Python 3.9+ |

        ---
        ### ⚙️ Detection Parameters (Sidebar)

        | Parameter | Effect |
        |---|---|
        | **Scale Factor** | Controls image pyramid step size. Lower value → more detections, slower |
        | **Min Neighbors** | Higher value → fewer false positives, may miss some faces |
        | **Min Face Size** | Smallest face (in pixels) to detect |

        ---
        ### 📁 Project Structure

        ```
        streamlit-app/
        ├── app.py               ← Main application (this file)
        ├── face_detector.py     ← Core detection logic
        ├── requirements.txt     ← Python dependencies
        └── README.md            ← Full project documentation
        ```

        ---
        ### 📄 License
        MIT — free to use and modify.
        """
    )
