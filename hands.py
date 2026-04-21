import cv2
import mediapipe as mp
import collections
import os
import urllib.request
import math

# --- 1. PREPARACIÓN DE MODELOS ---
def descargar_modelo(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

descargar_modelo("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", "hand_landmarker.task")
descargar_modelo("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", "face_landmarker.task")

# --- 2. CONFIGURACIÓN ---
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO, num_hands=1)

hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)

# --- 3. LÓGICA DE DETECCIÓN BASADA EN TU IMAGEN ---
def identificar_vocal(lm, lado):
    # Auxiliar para distancia euclidiana
    def dist(p1, p2):
        return math.sqrt((lm[p1].x - lm[p2].x)**2 + (lm[p1].y - lm[p2].y)**2)

    # --- DEFINICIONES SEGÚN TU LÓGICA ---
    # En MediaPipe, el eje Y crece hacia abajo. 
    # Por lo tanto: "Uña mayor que nudillo" en altura real significa PUNTA.y < NUDILLO.y
    
    punta_indice = lm[8].y
    nudillo_indice = lm[6].y
    
    punta_medio = lm[12].y
    nudillo_medio = lm[10].y
    
    punta_anular = lm[16].y
    nudillo_anular = lm[14].y
    
    punta_menique = lm[20].y
    nudillo_menique = lm[18].y

    # 1. NUEVA LÓGICA PARA LA "U" (según tu instrucción):
    # Índice y Meñique: Punta arriba (punta < nudillo)
    # Medio y Anular: Punta abajo (nudillo < punta)
    if (punta_indice < nudillo_indice and punta_menique < nudillo_menique) and \
       (nudillo_medio < punta_medio and nudillo_anular < punta_anular):
        return "U"

    # --- RESTO DE VOCALES (Manteniendo la estructura) ---
    
    # I: Solo meñique arriba
    if punta_menique < nudillo_menique and all(p > n for p, n in [(punta_indice, nudillo_indice), (punta_medio, nudillo_medio), (punta_anular, nudillo_anular)]):
        return "I"
    
    # A: Puño cerrado, pulgar lejos
    pulgar_lejos = dist(4, 9) > 0.08
    dedos_cerrados = all(p > n for p, n in [(punta_indice, nudillo_indice), (punta_medio, nudillo_medio), (punta_anular, nudillo_anular), (punta_menique, nudillo_menique)])
    if dedos_cerrados and pulgar_lejos and lm[4].y < lm[10].y:
        return "A"
    
    # E: Puño cerrado, puntas encogidas (no extendidas)
    if dedos_cerrados and lm[8].y > lm[6].y and not pulgar_lejos:
        return "E"
            
    # O: Círculo punta pulgar y punta índice
    if dist(4, 8) < 0.05 and nudillo_medio < punta_medio:
        return "O"

    return "Esperando..."

# --- 4. BUCLE PRINCIPAL ---
cap = cv2.VideoCapture(0)
vocal_detectada = "..."

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    ts = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    hand_result = hand_detector.detect_for_video(mp_image, ts)

    if hand_result.hand_landmarks:
        for idx, landmarks in enumerate(hand_result.hand_landmarks):
            lado = hand_result.handedness[idx][0].category_name
            vocal_detectada = identificar_vocal(landmarks, lado)

    # --- 5. UI ---
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (10, 10), (350, 150), (0, 0, 0), -1)
    cv2.putText(frame, "DETECTOR VOCALES", (20, 40), 1, 1.5, (255, 255, 255), 2)
    cv2.line(frame, (20, 60), (330, 60), (0, 255, 0), 2)
    cv2.putText(frame, f"LETRA: {vocal_detectada}", (20, 120), 1, 3, (0, 255, 0), 4)

    cv2.imshow('Vocales LSO', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()