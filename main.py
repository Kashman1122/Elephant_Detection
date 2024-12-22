from ultralytics import YOLO
import cvzone
import cv2
import math

# Running real-time from webcam
cap = cv2.VideoCapture(0)

model = YOLO('74.pt')

# Reading the classes
classnames = ['Elephant']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (320, 320))
    result = model(frame, stream=True)

    # Getting bbox, confidence, and class names information to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])

            # Detect class 0 or 1 (elephant or Elephant) with confidence greater than 75%
            if (Class == 0) and confidence > 80:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Red for elephant
                cvzone.putTextRect(frame, f'{classnames[Class]}', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
