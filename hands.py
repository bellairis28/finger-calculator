import cv2
import mediapipe as mp
import os
import urllib.request
import math

# Encapsulamos la lógica en una clase reutilizable para poder integrarla desde un servidor web.


def descargar_modelo(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)


class HandVowelDetector:
    """Detector de vocales basado en el código original.

    Uso:
        d = HandVowelDetector()
        letra = d.detect(frame_bgr)
        d.close()
    """

    def __init__(self):
        # Descargar modelos si no existen
        descargar_modelo(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            "hand_landmarker.task",
        )
        descargar_modelo(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            "face_landmarker.task",
        )

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
        )

        self.hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)

    def _identificar_vocal(self, lm):
        # Auxiliar para distancia euclidiana
        def dist(p1, p2):
            return math.sqrt((lm[p1].x - lm[p2].x) ** 2 + (lm[p1].y - lm[p2].y) ** 2)

        punta_indice = lm[8].y
        nudillo_indice = lm[6].y

        punta_medio = lm[12].y
        nudillo_medio = lm[10].y

        punta_anular = lm[16].y
        nudillo_anular = lm[14].y

        punta_menique = lm[20].y
        nudillo_menique = lm[18].y

        # U: Índice y meñique arriba, medio y anular abajo
        if (
            punta_indice < nudillo_indice
            and punta_menique < nudillo_menique
            and nudillo_medio < punta_medio
            and nudillo_anular < punta_anular
        ):
            return "U"

        # I: Solo meñique arriba
        if (
            punta_menique < nudillo_menique
            and all(
                p > n
                for p, n in [
                    (punta_indice, nudillo_indice),
                    (punta_medio, nudillo_medio),
                    (punta_anular, nudillo_anular),
                ]
            )
        ):
            return "I"

        pulgar_lejos = dist(4, 9) > 0.08
        dedos_cerrados = all(
            p > n
            for p, n in [
                (punta_indice, nudillo_indice),
                (punta_medio, nudillo_medio),
                (punta_anular, nudillo_anular),
                (punta_menique, nudillo_menique),
            ]
        )
        if dedos_cerrados and pulgar_lejos and lm[4].y < lm[10].y:
            return "A"

        if dedos_cerrados and lm[8].y > lm[6].y and not pulgar_lejos:
            return "E"

        # O: círculo entre pulgar e índice
        if dist(4, 8) < 0.05 and nudillo_medio < punta_medio:
            return "O"

        return "Esperando..."

    def detect(self, frame_bgr):
        """Recibe un frame en BGR (como entrega OpenCV), devuelve la vocal detectada (string)."""
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            hand_result = self.hand_detector.detect_for_video(mp_image, ts)

            if hand_result.hand_landmarks:
                # Solo procesamos la primera mano
                landmarks = hand_result.hand_landmarks[0]
                letra = self._identificar_vocal(landmarks)
                return letra

            return "Esperando..."
        except Exception:
            # No queremos que un error rompa el servidor; devolver estado neutro
            return "Error"

    def close(self):
        # Si hay limpieza necesaria, se podría hacer aquí. Por ahora no es requerido.
        pass


if __name__ == "__main__":
    # Mantener un pequeño ejemplo que no abre ventanas: captura y muestra la letra por consola.
    d = HandVowelDetector()
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            letra = d.detect(frame)
            print("LETRA:", letra)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        d.close()
