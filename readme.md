## 🧠 Lógica de Programación y Fundamentos Matemáticos

Este proyecto utiliza visión artificial avanzada para interpretar la **Lengua de Señas** mediante el análisis de la morfología de la mano en un espacio tridimensional.

### 1. Captura y Normalización de Coordenadas
El motor principal es **MediaPipe Hand Landmarker**, que identifica **21 puntos clave (landmarks)** de la mano. Cada punto posee coordenadas $(x, y, z)$ normalizadas en un rango de $[0.0, 1.0]$.
* **X e Y:** Representan la posición horizontal y vertical relativa al ancho y alto de la imagen.
* **Z:** Representa la profundidad, tomando como referencia la muñeca (punto 0).

### 2. Lógica Geométrica de los Dedos
Para determinar si un dedo está "abierto" o "cerrado", el algoritmo compara la posición de la **uña (tip)** con el **nudillo medio (PIP)**.

Dado que el origen $(0,0)$ de la cámara se encuentra en la esquina superior izquierda, la lógica matemática para detectar un dedo levantado es:
$$\text{Levantado} = \text{punta.y} < \text{nudillo.y}$$

Si el valor de $Y$ de la punta es menor, significa que está más cerca del borde superior de la pantalla (más alto físicamente).

---

### 3. Implementación de Vocales (Lógica Condicional)

El sistema evalúa el estado de los 5 dedos y sus distancias para clasificar la vocal:

#### **Vocal "A" (Puño cerrado con pulgar lateral)**
* **Matemática:** Se verifica que los 4 dedos largos tengan su $\text{punta.y} > \text{nudillo.y}$ (cerrados).
* **Pulgar:** Se mide la **Distancia Euclidiana** entre la punta del pulgar (4) y el nudillo del dedo medio (9).
    $$d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$$
* Si $d > 0.08$ (umbral de extensión), se confirma la "A".

#### **Vocal "E" (Puño cerrado con dedos encogidos)**
* Similar a la "A", pero con el pulgar contraído hacia la palma. Se valida que la punta del índice (8) esté por debajo de su segundo nudillo (6), creando un efecto de "garra" o puño plano.

#### **Vocal "I" (Solo meñique)**
* **Condición:** $\text{punta\_meñique.y} < \text{nudillo\_meñique.y}$ MIENTRAS QUE los demás dedos cumplen $\text{punta.y} > \text{nudillo.y}$.

#### **Vocal "O" (Morfología Circular)**
* Se basa exclusivamente en la proximidad de los extremos. Se activa cuando la distancia entre la punta del pulgar (4) y la punta del índice (8) es mínima:
    $$\text{distancia}(4, 8) < 0.05$$

#### **Vocal "U" (Lógica Específica de Usuario)**
* Según la configuración solicitada, la "U" se identifica mediante una combinación de extensión y contracción:
    * **Extendidos:** Índice (8) y Meñique (20) $\rightarrow (\text{punta.y} < \text{nudillo.y})$.
    * **Contraídos:** Medio (12) y Anular (16) $\rightarrow (\text{nudillo.y} < \text{punta.y})$.

---

### 4. Procesamiento en Tiempo Real
El script utiliza un **Bucle de Eventos** que procesa cada frame de la webcam:
1.  **Conversión de Color:** De BGR (OpenCV) a RGB (MediaPipe).
2.  **Timestamping:** Cada frame recibe un sello de tiempo en milisegundos ($ts$) para que el modelo mantenga la coherencia en el flujo de video (`RunningMode.VIDEO`).
3.  **Flip Horizontal:** Se aplica un efecto espejo (`cv2.flip`) para que el usuario pueda interactuar de forma intuitiva (si mueve la mano a su derecha, se mueve a la derecha en pantalla).

### 5. Interfaz de Usuario (UI)
Se implementa una **Máquina de Estados** simple donde la variable `vocal_detectada` se actualiza en cada ciclo y se renderiza sobre el frame utilizando `cv2.putText`, permitiendo una retroalimentación visual instantánea con una latencia mínima.