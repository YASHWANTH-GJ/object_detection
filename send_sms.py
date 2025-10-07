import cv2
import time
import numpy as np
import imutils
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuration
SMTP_SERVER = "smtp.gmail.com"  # Gmail SMTP server
SMTP_PORT = 587  # Gmail SMTP port
EMAIL_ADDRESS = "your mail"  # Your email address
EMAIL_PASSWORD = "google app password"  # Gmail app password (if using 2FA)
PHONE_NUMBER = "your number"  # Recipient's phone number
CARRIER_GATEWAY = "vtext.com"  # Carrier's SMS gateway domain (Verizon example)
SMS_GATEWAY = f"{PHONE_NUMBER}@{CARRIER_GATEWAY}"  # Complete SMS gateway address

def send_sms_via_email(message):
    try:
        # Create the message container
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = SMS_GATEWAY
        msg['Subject'] = "Alert"
        
        # Attach the message content
        body = MIMEText(message, 'plain')
        msg.attach(body)
        
        # Connect to the SMTP server
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Start TLS encryption
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)  # Login to the SMTP server
            server.sendmail(EMAIL_ADDRESS, SMS_GATEWAY, msg.as_string())  # Send the email
        print("[INFO] Message sent successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to send message: {e}")

# Load YOLO model and COCO labels
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize webcam
cam = cv2.VideoCapture(0)
time.sleep(1)

firstFrame = None
area = 500
alert_sent = False  # To prevent spamming alerts

while True:
    _, img = cam.read()
    text = "Normal"
    img = imutils.resize(img, width=500)
    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)
    
    if firstFrame is None:
        firstFrame = gaussianImg
        continue

    imgDiff = cv2.absdiff(firstFrame, gaussianImg)
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)
    
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object detected"

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
            if confidence > 0.5:
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
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if label == 'cat' and not alert_sent:
                send_sms_via_email("catdetected")
                alert_sent = True  # Prevent multiple alerts
    
    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("cameraFeed", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

