import cv2
import time
import os
import requests
from datetime import datetime

# =======================
# CONFIG
# =======================
# 👉 Replace these with your real values
TELEGRAM_BOT_TOKEN = "your_token"
TELEGRAM_CHAT_ID = "your_token_id"  # e.g. "123456789"

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

DETECTION_COOLDOWN = 15  

# =======================
# TELEGRAM FUNCTIONS
# =======================
def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text
    }
    try:
        response = requests.post(url, data=data, timeout=5)
        if not response.ok:
            print("Failed to send Telegram message:", response.text)
    except Exception as e:
        print("Error sending Telegram message:", e)

def send_telegram_photo(photo_path: str, caption: str = ""):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(photo_path, "rb") as photo_file:
            files = {"photo": photo_file}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            response = requests.post(url, files=files, data=data, timeout=10)
            if not response.ok:
                print("Failed to send Telegram photo:", response.text)
    except Exception as e:
        print("Error sending Telegram photo:", e)


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    last_alert_time = 0

    print("Starting Smart Intrusion Detector...")
    send_telegram_message("🔐 Smart Intrusion Detector started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam")
            break

        frame_resized = cv2.resize(frame, (640, 480))

        (rects, weights) = hog.detectMultiScale(
            frame_resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )

        for (x, y, w, h) in rects:
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # If at least one person is detected
        if len(rects) > 0:
            current_time = time.time()
            if current_time - last_alert_time > DETECTION_COOLDOWN:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                snapshot_path = os.path.join(SNAPSHOT_DIR, f"intrusion_{timestamp}.jpg")
                cv2.imwrite(snapshot_path, frame_resized)

                alert_text = f"🚨 Intrusion detected at {timestamp}"
                print(alert_text)

                # Send alerts
                send_telegram_message(alert_text)
                send_telegram_photo(snapshot_path, caption=alert_text)

                last_alert_time = current_time

        cv2.imshow("Smart Intrusion Detector - Press Q to quit", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    send_telegram_message("🛑 Smart Intrusion Detector stopped.")

if __name__ == "__main__":
    main()
