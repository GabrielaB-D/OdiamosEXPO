import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Variables globales para guardar resultados
last_result = None

# Callback que guarda resultados y muestra cuántas manos detectó
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global last_result
    last_result = result
    print(f"[{timestamp_ms}] Manos detectadas: {len(result.hand_landmarks)}")

# Configuración del detector
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),  # pon la ruta a tu modelo
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result
)

# Captura de video (webcam)
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir frame a RGB para mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Timestamp en milisegundos
        timestamp_ms = int(time.time() * 1000)

        # Procesar frame de manera asíncrona
        landmarker.detect_async(mp_image, timestamp_ms)

        # Dibujar los landmarks si hay resultados
        if last_result and last_result.hand_landmarks:
            for hand_landmarks in last_result.hand_landmarks:
                # Dibujar conexiones entre los landmarks (basadas en el modelo de MediaPipe Hands)
                connections = mp.solutions.hands.HAND_CONNECTIONS
                h, w, _ = frame.shape
                points = []

                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                # Dibujar líneas entre puntos
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(points) and end_idx < len(points):
                        cv2.line(frame, points[start_idx], points[end_idx], (0, 150, 255), 2)

        # Mostrar la ventan5
        cv2.imshow("Hand Landmarker", frame)

        # Salir con ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
