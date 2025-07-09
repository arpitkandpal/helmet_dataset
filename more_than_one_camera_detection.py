import cv2
import time
import threading
from datetime import datetime
import smtplib
from email.message import EmailMessage
import os
import geocoder
import pandas as pd
from ultralytics import YOLO

# ======= CONFIGURATION =======
SENDER_EMAIL = "arpitkandpal10b@gmail.com"
APP_PASSWORD = "gmxptegpte"
RECEIVER_EMAIL = "arpit.kandpal2025@gmail.com"
EXCEL_FILE = "detections_log.xlsx"
CAPTURE_DIR = "alerts"
ALERT_DELAY_SECONDS = 30  # Wait this much before sending another alert from same camera

# Camera sources: 0 = laptop webcam, next = DroidCam stream or other IP camera
CAMERA_SOURCES = [
    0,  # Laptop
    "https://192.168.29.149:4343/video",  # Phone camera via DroidCam (change IP if needed)
]

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Ensure alert folder exists
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Track last alert time for each camera
last_alert_time = {}

def send_email_and_log(image_path, timestamp, location):
    try:
        # ========== Send Email ==========
        msg = EmailMessage()
        msg["Subject"] = "No Helmet Detected!"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg.set_content(f"No helmet detected!\nTime: {timestamp}\nLocation: {location}")

        with open(image_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename=os.path.basename(image_path))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        print("[INFO] Email sent.")

        # ========== Log to Excel ==========
        if os.path.exists(EXCEL_FILE):
            df = pd.read_excel(EXCEL_FILE)
        else:
            df = pd.DataFrame(columns=["Date", "Time", "Location"])

        date_str = timestamp.split()[0]
        time_str = timestamp.split()[1]
        df.loc[len(df)] = [date_str, time_str, location]
        df.to_excel(EXCEL_FILE, index=False)
        print("[INFO] Logged to Excel.")
    except Exception as e:
        print("[ERROR] Alert failed:", e)

def detect_from_camera(cam_id, source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Camera {cam_id} could not be opened.")
        return

    print(f"[INFO] Camera {cam_id} started")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Camera {cam_id} failed to read.")
            break

        results = model(frame, verbose=False)[0]
        current_time = time.time()

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw rectangle
            color = (0, 0, 255) if label == "nohelmet" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Check for "nohelmet" label
            if label == "nohelmet":
                last_time = last_alert_time.get(cam_id, 0)
                if current_time - last_time > ALERT_DELAY_SECONDS:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    location = geocoder.ip("me").address or "Unknown Location"
                    filename = os.path.join(CAPTURE_DIR, f"cam{cam_id}_nohelmet_{int(current_time)}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"[ALERT] Camera {cam_id} - No helmet detected.")
                    last_alert_time[cam_id] = current_time

                    threading.Thread(target=send_email_and_log, args=(filename, timestamp, location)).start()

        # Show frame
        try:
            cv2.imshow(f"Camera {cam_id}", frame)
        except cv2.error:
            pass  # Ignore broken windows

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    try:
        cv2.destroyWindow(f"Camera {cam_id}")
    except cv2.error:
        pass

# ===== START THREAD FOR EACH CAMERA =====
for i, src in enumerate(CAMERA_SOURCES):
    threading.Thread(target=detect_from_camera, args=(i, src)).start()

