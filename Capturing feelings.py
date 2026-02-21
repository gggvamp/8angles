

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:06:29 2026

@author: Gerardo
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:06:29 2026
@author: Gerardo
TOP-8 Facial Angular Descriptors
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import os

# ===============================
# CONFIGURACIÓN
# ===============================
archivo_excel = "dataset_emociones_angulares_top8.xlsx"

emociones_teclas = {
    ord('h'): "happy",
    ord('a'): "angry",
    ord('s'): "sad",
    ord('u'): "surprise",
    ord('f'): "fear",
    ord('n'): "neutral"
}

# ===============================
# LANDMARKS (27 puntos)
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
# ÁNGULOS SELECCIONADOS (TOP-8)
# ===============================
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
# COLORES POR ÁNGULO
# ===============================
ANGLE_COLORS = {
    "θ1": (255, 0, 0),
    "θ2": (0, 255, 0),
    "θ3": (0, 0, 255),
    "θ4": (255, 255, 0),
    "θ5": (255, 0, 255),
    "θ6": (0, 255, 255),
    "θ7": (128, 0, 255),
    "θ8": (255, 128, 0)
}

# ===============================
# FUNCIONES
# ===============================
def calcular_angulo(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    ang = math.degrees(math.atan2(np.linalg.det([a, b]), np.dot(a, b)))
    return ang + 360 if ang < 0 else ang

def obtener_puntos(face_landmarks, w, h):
    return {
        k: (
            int(face_landmarks[mp_id].x * w),
            int(face_landmarks[mp_id].y * h)
        )
        for k, mp_id in KEY_LANDMARKS.items()
    }

def calcular_angulos(puntos):
    return {
        nombre: calcular_angulo(puntos[i], puntos[j], puntos[k])
        for nombre, (i, j, k) in ANGLE_DEFS.items()
    }

# ===============================
# INICIALIZACIÓN MEDIAPIPE
# ===============================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1
)

cap = cv2.VideoCapture(0)

# ===============================
# DATASET EN MEMORIA
# ===============================
columnas = ["emocion"] + list(ANGLE_DEFS.keys())

if os.path.exists(archivo_excel):
    df = pd.read_excel(archivo_excel)
else:
    df = pd.DataFrame(columns=columnas)

grabando = False
emocion_actual = None

print("h=happy | a=angry | s=sad | u=surprise | f=fear | n=neutral")
print("ESC = salir")

# ===============================
# LOOP PRINCIPAL
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
        puntos = obtener_puntos(face.landmark, w, h)
        angulos = calcular_angulos(puntos)
        
        
        

        # Dibujar ángulos
        for nombre, (i, j, k) in ANGLE_DEFS.items():
            color = ANGLE_COLORS[nombre]
            cv2.circle(frame, puntos[i], 4, color, -1)
            cv2.circle(frame, puntos[j], 5, (255, 255, 255), -1)
            cv2.circle(frame, puntos[k], 4, color, -1)
            cv2.line(frame, puntos[j], puntos[i], color, 1)
            cv2.line(frame, puntos[j], puntos[k], color, 1)

        # Texto
        for idx, (nombre, valor) in enumerate(angulos.items()):
            cv2.putText(
                frame,
                f"{nombre}: {valor:.1f}",
                (10, 30 + idx * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                ANGLE_COLORS[nombre],
                2
            )

        # Guardado incremental
        if grabando:
            df.loc[len(df)] = [emocion_actual] + list(angulos.values())

    cv2.imshow("Emotion Capture – Angular Descriptors (Top-8)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in emociones_teclas:
        emocion_actual = emociones_teclas[key]
        grabando = True
        print(f">>> CAPTURANDO EMOCIÓN: {emocion_actual.upper()}")

    if key == 27:
        break

# ===============================
# CIERRE
# ===============================
cap.release()
cv2.destroyAllWindows()
df.to_excel(archivo_excel, index=False)

print("Dataset guardado correctamente en:", archivo_excel)
