# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 21:08:52 2026

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
TEMPORAL_WINDOW = 10

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

ANGLE_DEFS = {
    "θ1": (0, 2, 1),
    "θ2": (0, 7, 1),
    "θ3": (0, 7, 8),
    "θ4": (2, 1, 10),
    "θ5": (3, 0, 7),
    "θ6": (13, 12, 14),
    "θ7": (11, 13, 12),
    "θ8": (21, 12, 14)
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
# 1. LOAD DATASET & TRAIN
# ===============================
print("[INFO] Loading dataset...")

data = pd.read_excel(DATASET_PATH)

# 🔥 ADAPTACIÓN IMPORTANTE
feature_columns = [col for col in data.columns if col.startswith("θ")]

X = data[feature_columns].values
y = data["emocion"].values

print("[INFO] Features used:", feature_columns)

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
# 2. MEDIAPIPE
# ===============================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1
)

cap = cv2.VideoCapture(0)

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

        avg_probs = np.mean(prob_buffer, axis=0)
        pred_idx = np.argmax(avg_probs)
        emotion = class_names[pred_idx]
        confidence = avg_probs[pred_idx]

        cv2.putText(
            frame,
            f"Emotion: {emotion.upper()} ({confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # FPS
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

cap.release()
cv2.destroyAllWindows()
print("[INFO] System terminated correctly.")