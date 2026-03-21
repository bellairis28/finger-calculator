# Calculadora IA con MediaPipe

## Instalación rápida
1. Clonar el repositorio: `git clone https://github.com/justlebadura/finger-calculator`
2. Crear entorno virtual: `python -m venv env`
3. Activar entorno:
   - Windows: `.\env\Scripts\activate`
   - Linux/Mac: `source env/bin/activate`
4. Instalar dependencias: `pip install -r requirements.txt`
5. Ejecutar: `python calculadora.py`

## Referencia de Landmarks (Puntos de la Mano)

El sistema utiliza el modelo **MediaPipe Hand Landmarker**, que identifica 21 puntos clave (0-20) en la mano. Para la lógica de la calculadora, nos enfocamos en las puntas (Tips) y los nudillos (PIP/MCP).

### Tabla de Variables y Puntos Clave

| Dedo | Punto Punta (Tip) | Punto Referencia (PIP/MCP) | Eje de Medición |
| :--- | :---: | :---: | :---: |
| **Pulgar** | 4 | 5 (Nudillo Índice) | Eje X (Horizontal) |
| **Índice** | 8 | 6 | Eje Y (Vertical) |
| **Corazón** | 12 | 10 | Eje Y (Vertical) |
| **Anular** | 16 | 14 | Eje Y (Vertical) |
| **Meñique** | 20 | 18 | Eje Y (Vertical) |

### Lógica de Detección
- **Dedos 2-5:** Se consideran "levantados" si la coordenada `y` del **Tip** es menor que la del punto de referencia (recordando que en OpenCV el 0 está arriba).
- **Pulgar:** Se considera "levantado" mediante una comparación en el eje `x` respecto al nudillo del corazón o índice, adaptado a la lateralidad (Mano Izquierda vs. Derecha).