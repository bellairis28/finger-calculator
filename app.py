from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
from hands import HandVowelDetector

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev"
socketio = SocketIO(app, cors_allowed_origins="*")

# Detector global (se crea al iniciar el servidor)
detector = HandVowelDetector()


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("frame")
def handle_frame(data):
    # data: data:image/jpeg;base64,...
    img_b64 = data.split(",", 1)[1]
    img_bytes = base64.b64decode(img_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    letra = detector.detect(frame)
    emit("letra", {"letra": letra})


@socketio.on("connect")
def connect():
    print("Cliente conectado")


@socketio.on("disconnect")
def disconnect():
    print("Cliente desconectado")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
