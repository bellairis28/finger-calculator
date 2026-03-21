import cv2
import mediapipe as mp
import collections
import os
import urllib.request

# --- 1. PREPARACIÓN DE MODELOS ---
def descargar_modelo(url, path):
    if not os.path.exists(path):
        print(f"Descargando {path}...")
        urllib.request.urlretrieve(url, path)

descargar_modelo("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", "hand_landmarker.task")
descargar_modelo("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", "face_landmarker.task")

# --- 2. CONFIGURACIÓN DE MEDIAPIPE ---
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configuración Manos
hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO, num_hands=2)

# Configuración Cara (Con Blendshapes)
face_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO, output_face_blendshapes=True)

hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
face_detector = mp.tasks.vision.FaceLandmarker.create_from_options(face_options)

# --- 3. LÓGICA DE CONTEO (HANDS) ---
def contar_dedos(landmarks, lado):
    dedos = [1 if landmarks[i].y < landmarks[i-2].y else 0 for i in [8, 12, 16, 20]]
    # Pulgar (Lógica invertida según tu calibración)
    px, rx = landmarks[4].x, landmarks[5].x
    dedos.append(1 if (lado == "Right" and px > rx) or (lado == "Left" and px < rx) else 0)
    return dedos.count(1)

# --- 4. BUCLE PRINCIPAL ---
cap = cv2.VideoCapture(0)
historial_num = collections.deque(maxlen=5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Preparación de imagen
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    ts = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    # DETECCIÓN DOBLE
    hand_result = hand_detector.detect_for_video(mp_image, ts)
    face_result = face_detector.detect_for_video(mp_image, ts)

    # Lógica de Manos (Números)
    n1, n2 = 0, 0
    if hand_result.hand_landmarks:
        for idx, landmarks in enumerate(hand_result.hand_landmarks):
            lado = hand_result.handedness[idx][0].category_name
            if lado == "Left": n1 = contar_dedos(landmarks, lado)
            else: n2 = contar_dedos(landmarks, lado)

    # Lógica de Cara (Operaciones)
    op_label = "ESPERANDO GESTO..."
    resultado = 0
    color_op = (255, 255, 255)

    if face_result.face_blendshapes:
        # Extraemos los scores de los gestos
        scores = {category.category_name: category.score for category in face_result.face_blendshapes[0]}
        
        # Umbrales de detección (sensibilidad)
        if scores.get("browOuterUpLeft", 0) > 0.3:
            op_label, resultado, color_op = "SUMA (+)", n1 + n2, (0, 255, 0)
        elif scores.get("browOuterUpRight", 0) > 0.3:
            op_label, resultado, color_op = "RESTA (-)", n1 - n2, (0, 0, 255)
        elif scores.get("eyeBlinkLeft", 0) > 0.5:
            op_label, resultado, color_op = "MULTIPLICA (x)", n1 * n2, (255, 255, 0)
        elif scores.get("eyeBlinkRight", 0) > 0.5:
            op_label, resultado, color_op = "DIVIDE (/)", (n1 / n2 if n2 != 0 else "Error"), (0, 255, 255)

    # --- 5. INTERFAZ DE USUARIO (UI) ---
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (10, 10), (350, 180), (0, 0, 0), -1)
    
    cv2.putText(frame, f"N1 (Izq): {n1} | N2 (Der): {n2}", (20, 40), 1, 1.2, (255,255,255), 2)
    cv2.putText(frame, f"OPERACION:", (20, 80), 1, 1, (200,200,200), 1)
    cv2.putText(frame, op_label, (20, 110), 1, 1.5, color_op, 2)
    
    cv2.line(frame, (20, 125), (330, 125), (100,100,100), 1)
    cv2.putText(frame, f"RESULTADO: {resultado}", (20, 160), 1, 1.8, (0, 255, 0), 3)

    cv2.imshow('Calculadora Facial-Manual Sapere', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()