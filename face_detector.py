"""
face_detector.py
----------------
Core face detection module using OpenCV's Haar Cascade Classifier.
Provides functions for detecting faces in images and drawing bounding boxes.
"""

import cv2
import numpy as np
from PIL import Image
import os


def load_cascade():
    """
    Load the Haar Cascade classifier for frontal face detection.
    Uses OpenCV's built-in data path — no hardcoded or external file required.

    Returns:
        cv2.CascadeClassifier: Loaded cascade classifier.

    Raises:
        RuntimeError: If the cascade file cannot be loaded.
    """
    # Use OpenCV's bundled haarcascade — avoids any hardcoded paths
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    if not os.path.exists(cascade_path):
        raise RuntimeError(
            f"Haar Cascade file not found at: {cascade_path}\n"
            "Please ensure opencv-python-headless is installed correctly."
        )

    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        raise RuntimeError("Failed to load Haar Cascade classifier.")

    return cascade


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to an OpenCV BGR numpy array.

    Args:
        pil_image (PIL.Image.Image): Input PIL image.

    Returns:
        np.ndarray: OpenCV-compatible BGR image array.
    """
    # Convert PIL RGBA/P images to RGB first for consistency
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")

    # PIL uses RGB; OpenCV uses BGR — flip the channel order
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert an OpenCV BGR numpy array to a PIL Image.

    Args:
        cv2_image (np.ndarray): OpenCV BGR image array.

    Returns:
        PIL.Image.Image: Converted PIL image.
    """
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def detect_faces(
    pil_image: Image.Image,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple = (30, 30),
) -> dict:
    """
    Detect faces in a PIL image and return detection results.

    Args:
        pil_image (PIL.Image.Image): Input image.
        scale_factor (float): How much the image size is reduced at each scale (>1.0).
        min_neighbors (int): How many neighbors each candidate rectangle should retain.
        min_size (tuple): Minimum possible object size (width, height).

    Returns:
        dict: {
            "faces": list of (x, y, w, h) tuples for each detected face,
            "count": number of faces detected,
            "image_size": (width, height) of original image,
            "processed_image": PIL Image with bounding boxes drawn,
            "confidence_scores": list of approximate confidence values,
        }
    """
    cascade = load_cascade()

    # Convert to OpenCV format
    cv_img = pil_to_cv2(pil_image)

    # Convert to grayscale for detection (Haar cascades work on grayscale)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve detection in varied lighting
    gray = cv2.equalizeHist(gray)

    # Detect faces using the cascade classifier
    # detectMultiScale3 returns faces + reject levels for confidence approximation
    faces, _, weights = cascade.detectMultiScale3(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        outputRejectLevels=True,
    )

    face_list = []
    confidence_scores = []

    if len(faces) > 0:
        for face, weight in zip(faces, weights):
            face_list.append(tuple(face))
            # Normalize weight to a 0–100 "confidence" display value
            score = min(100, round(float(weight) * 10, 1))
            confidence_scores.append(score)

    # Draw bounding boxes on a copy of the original image
    annotated = cv_img.copy()
    for x, y, w, h in face_list:
        # Draw green rectangle around each detected face
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw a small filled circle at top-left corner of each box
        cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)

    processed_pil = cv2_to_pil(annotated)
    width, height = pil_image.size

    return {
        "faces": face_list,
        "count": len(face_list),
        "image_size": (width, height),
        "processed_image": processed_pil,
        "confidence_scores": confidence_scores,
    }


def draw_labeled_boxes(
    pil_image: Image.Image,
    faces: list,
    confidence_scores: list,
    show_labels: bool = True,
) -> Image.Image:
    """
    Draw labeled bounding boxes with face index and confidence on the image.

    Args:
        pil_image (PIL.Image.Image): Original image.
        faces (list): List of (x, y, w, h) face bounding boxes.
        confidence_scores (list): Corresponding confidence scores (0–100).
        show_labels (bool): Whether to render label text above each box.

    Returns:
        PIL.Image.Image: Annotated image with labeled boxes.
    """
    cv_img = pil_to_cv2(pil_image)

    for idx, ((x, y, w, h), score) in enumerate(zip(faces, confidence_scores), start=1):
        # Main bounding box — bright green
        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if show_labels:
            label = f"Face {idx}  {score}%"
            label_y = y - 10 if y - 10 > 10 else y + h + 20

            # Semi-transparent label background
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(
                cv_img,
                (x, label_y - lh - 4),
                (x + lw + 4, label_y + 2),
                (0, 200, 0),
                cv2.FILLED,
            )

            # White text over the green background
            cv2.putText(
                cv_img,
                label,
                (x + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return cv2_to_pil(cv_img)
