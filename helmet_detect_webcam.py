import cv2
import time
from datetime import datetime
import smtplib
from email.message import EmailMessage
import os
import geocoder
import threading
import pandas as pd
from ultralytics import YOLO

# ======= CONFIGURATION =======
SENDER_EMAIL = "arpitkandpal10b@gmail.com"
APP_PASSWORD = "gmxpttegpte"
RECEIVER_EMAIL = "arpit.kandpal2025@gmail.com"
EXCEL_FILE = "detections_log.xlsx"
CAPTURE_DIR = "alerts"
RTSP_URL = "rtsp://localhost:8554/test"  # Replace with your actual RTSP URL

# Print RTSP ID
print(f"[INFO] Using RTSP Stream: {RTSP_URL}")

# Create alerts directory if not exists
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Load YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")

# Function to send email and log data
def send_email_and_log(image_path, timestamp, location):
    try:
        # Send email
        msg = EmailMessage()
        msg["Subject"] = "No Helmet Detected!"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg.set_content(f"No helmet detected!\n\nTime: {timestamp}\nLocation: {location}")

        with open(image_path, "rb") as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=os.path.basename(image_path))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        print("[INFO] Email sent.")

        # Log to Excel
        if os.path.exists(EXCEL_FILE):
            df = pd.read_excel(EXCEL_FILE)
        else:
            df = pd.DataFrame(columns=["Date", "Time", "Location"])

        date_str, time_str = timestamp.split()
        df.loc[len(df)] = [date_str, time_str, location]
        df.to_excel(EXCEL_FILE, index=False)
        print("[INFO] Logged to Excel.")
    
    except Exception as e:
        print("[ERROR] Failed to send email or log to Excel:", e)

# ===== MAIN DETECTION LOOP =====
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    print("[ERROR] Failed to open RTSP stream.")
    exit()

print("[INFO] RTSP stream opened successfully.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read from stream.")
        break

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255) if label == "nohelmet" else (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # If 'nohelmet' detected
        if label == "nohelmet":
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            location = geocoder.ip("me").address or "Unknown Location"
            filename = f"{CAPTURE_DIR}/nohelmet_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)

            print("[INFO] No helmet detected. Starting alert thread...")
            threading.Thread(target=send_email_and_log, args=(filename, timestamp, location)).start()

    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



