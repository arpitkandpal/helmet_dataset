
from ultralytics import YOLO
import cv2

# Load model on GPU
model = YOLO("best.pt").to("cuda")

# Open laptop webcam
cap = cv2.VideoCapture(0)  # Use RTSP URL here if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference on GPU
    results = model.predict(source=frame, device='cuda', imgsz=640, conf=0.5)

    # Annotated output
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("Helmet Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

