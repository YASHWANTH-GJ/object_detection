import cv2
import time
import numpy as np
from playsound import playsound
from twilio.rest import Client
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Twilio configuration
ACCOUNT_SID = 'AC4331b3eeac3ee52a3ca6cf6f02cc7d2e'
AUTH_TOKEN = '6cbcd87788d0704317e3327234c84bea'
TWILIO_PHONE_NUMBER = '+12183570669'
RECIPIENT_PHONE_NUMBER = '+919241372883'

# Email configuration
EMAIL_ADDRESS = 'yashwanthgj7@gmail.com'
EMAIL_PASSWORD = 'hdsc imkk qqdl pyeb'
TO_EMAIL_ADDRESS = 'yashwanthgj2@gmail.com'

def send_sms_via_twilio(message):
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE_NUMBER
        )
        logging.info(f"SMS sent: {message}")
    except Exception as e:
        logging.error(f"Failed to send SMS: {e}")

def send_email_with_image(subject, body, to_email, image):
    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email

        # Attach the text body
        msg.attach(MIMEText(body, 'plain'))

        # Attach the image
        if image is not None:
            img = MIMEImage(image, name='snapshot.jpg')
            msg.attach(img)

        logging.debug("Connecting to SMTP server...")
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.set_debuglevel(1)  # Enable debug output
            logging.debug("Starting TLS...")
            server.starttls()
            logging.debug("Logging in...")
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            logging.debug("Sending email...")
            server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
        logging.info("Email sent.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

# Load YOLO model and COCO labels
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize webcam
cam = cv2.VideoCapture(0)
time.sleep(1)

alert_timeout = 60  # Alert cooldown in seconds
last_alert_time = time.time()

def play_alarm():
    """Play alarm sound."""
    try:
        playsound("giraffe_alarm.mp3")
    except Exception as e:
        logging.error(f"Failed to play alarm: {e}")

while True:
    ret, img = cam.read()
    if not ret:
        logging.error("Failed to capture frame.")
        break

    # YOLO Object Detection
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:  # Increased confidence threshold for better accuracy
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label == "giraffe":
                play_alarm()
            elif label in ["cat", "cow"] and (time.time() - last_alert_time) > alert_timeout:
                ret, buffer = cv2.imencode('.jpg', img)
                if ret:
                    subject = f"{label.capitalize()} Detected"
                    body = f"A {label} has been detected in the camera feed."
                    send_sms_via_twilio(f"{label.capitalize()} detected in your camera.")
                    send_email_with_image(subject, body, TO_EMAIL_ADDRESS, buffer.tobytes())
                last_alert_time = time.time()

    cv2.imshow("Camera Feed", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

