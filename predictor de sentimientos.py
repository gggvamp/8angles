# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 14:11:25 2026

@author: Gerardo
"""

# -*- coding: utf-8 -*-
"""
Real-Time Facial Emotion Recognition
TOP-8 Angular Descriptors + MLP (Auto-trained)

Author: Gerardo García-Gil
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import time

from collections import deque

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# ===============================
# CONFIGURATION
# ===============================
DATASET_PATH = "dataset_emociones_angulares_top8.xlsx"
TEMPORAL_WINDOW = 10   # number of frames for temporal smoothing

# ===============================
# LANDMARKS (27 points)
# ===============================
KEY_LANDMARKS = {
    0: 61,   1: 292,  2: 0,   3: 17,
    4: 50,   5: 280,  6: 48,  7: 4,
    8: 289,  9: 206, 10: 426,
    11: 133, 12: 130, 13: 159, 14: 145,
    15: 362, 16: 359, 17: 386, 18: 374,
    19: 122, 20: 351,
    21: 46,  22: 105, 23: 107,
    24: 276, 25: 334, 26: 336
}

# ===============================
# TOP-8 ANGULAR DESCRIPTORS
# ===============================
ANGLE_DEFS = {
    "θ1": (0, 2, 1),    # Central mouth (AU25–26)
    "θ2": (0, 7, 1),    # Mouth–nose (AU9–10)
    "θ3": (0, 7, 8),    # Lip corners (AU12–14)
    "θ4": (2, 1, 10),   # Mouth–cheek
    "θ5": (3, 0, 7),    # Nose wrinkler (AU10)
    "θ6": (13, 12, 14), # Eye aperture (AU5–7)
    "θ7": (11, 13, 12), # Eyebrow raise (AU1–2)
    "θ8": (21, 12, 14)  # Brow lowerer (AU4)
}

# ===============================
# FUNCTIONS
# ===============================
def calculate_angle(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    angle = math.degrees(math.atan2(np.linalg.det([a, b]), np.dot(a, b)))
    return angle + 360 if angle < 0 else angle

def extract_points(face_landmarks, w, h):
    return {
        k: (
            int(face_landmarks[mp_id].x * w),
            int(face_landmarks[mp_id].y * h)
        )
        for k, mp_id in KEY_LANDMARKS.items()
    }

def compute_angles(points):
    return np.array([
        calculate_angle(points[i], points[j], points[k])
        for (i, j, k) in ANGLE_DEFS.values()
    ])

# ===============================
# 1. LOAD DATASET & TRAIN MLP
# ===============================
print("[INFO] Loading dataset and training MLP...")

data = pd.read_excel(DATASET_PATH)
X = data.drop(columns=["emocion"]).values
y = data["emocion"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = le.classes_

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        random_state=42
    ))
])

pipe.fit(X, y_enc)

print("[INFO] Model trained successfully.")
print("[INFO] Classes:", class_names)

# ===============================
# 2. MEDIAPIPE INITIALIZATION
# ===============================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1
)

cap = cv2.VideoCapture(0)

# ===============================
# TEMPORAL BUFFERS
# ===============================
prob_buffer = deque(maxlen=TEMPORAL_WINDOW)

prev_time = time.time()

# ===============================
# 3. REAL-TIME LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        points = extract_points(face.landmark, w, h)

        angles = compute_angles(points).reshape(1, -1)

        probs = pipe.predict_proba(angles)[0]
        prob_buffer.append(probs)

        # Temporal smoothing
        avg_probs = np.mean(prob_buffer, axis=0)
        pred_idx = np.argmax(avg_probs)
        emotion = class_names[pred_idx]
        confidence = avg_probs[pred_idx]

        # ===============================
        # DISPLAY
        # ===============================
        y0 = 30
        cv2.putText(
            frame,
            f"Emotion: {emotion.upper()} ({confidence:.2f})",
            (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        for i, (cls, p) in enumerate(zip(class_names, avg_probs)):
            cv2.putText(
                frame,
                f"{cls}: {p:.2f}",
                (10, y0 + 30 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                1
            )

    # ===============================
    # FPS
    # ===============================
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 120, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow("Real-Time Facial Emotion Recognition (Top-8)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ===============================
# 4. CLEAN EXIT
# ===============================
cap.release()
cv2.destroyAllWindows()
print("[INFO] System terminated correctly.")
