import cv2
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
import pytz
from super_gradients.training import models
from PIL import Image
import tempfile
import os
import sys
import time
load_dotenv()

# Load your custom YOLO-NAS-M model
yolo_nas_m = models.get('yolo_nas_m', num_classes=2, checkpoint_path="ckpt_best.pth")
# yolo_nas_m2 = models.get("yolo_nas_m", pretrained_weights='coco')

# Load YOLO-V4 model
# yolo_v4 = models.get('yolo_nas_m', num_classes=2, checkpoint_path="ckpt_best3.pth")

# Load YOLO-NAS-M Premium model
# yolo_nas_m_premium = models.get('yolo_nas_m', num_classes=2, checkpoint_path="ckpt_best8.pth")

# Define the class index for elephant (class index 0 in custom model)
elephant_class_index = 0
elephant_class_index2 = 20

# Define the confidence threshold
confidence_threshold = 0.75

def draw_bounding_boxes(frame, pred_data):
    num_elephants = 0
    for i in range(len(pred_data.labels)):
        label = pred_data.labels[i]
        confidence = pred_data.confidence[i]

        if label == elephant_class_index and confidence > confidence_threshold:
            bbox = pred_data.bboxes_xyxy[i]  # Assuming the bounding boxes are stored in bboxes_xyxy
            x1, y1, x2, y2 = map(int, bbox)

            num_elephants += 1

            # Draw bounding box and label for the elephant
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Elephant {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # if label == elephant_class_index2 and confidence > confidence_threshold:
        #     bbox = pred_data.bboxes_xyxy[i]  # Assuming the bounding boxes are stored in bboxes_xyxy
        #     x1, y1, x2, y2 = map(int, bbox)

            # num_elephants += 1
            #
            # # Draw bounding box and label for the elephant
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, f'Elephant {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return num_elephants

# Function to perform real-time detection from webcam using multiple YOLO models
def detect_with_multiple_models():
    # Open webcam or video file
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Apply bilateral filtering to reduce noise
        filtered_frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

        # Perform prediction on the filtered frame using YOLO-NAS-M model
        predictions_nas_m = yolo_nas_m.predict(filtered_frame)
        # predictions_nas_m2 = yolo_nas_m2.predict(filtered_frame)

        # Perform prediction on the filtered frame using YOLO-V4 model
        # predictions_v4 = yolo_v4.predict(filtered_frame)
        # Perform prediction on the filtered frame using YOLO-NAS-M Premium model
        # predictions_nas_m_premium = yolo_nas_m_premium.predict(filtered_frame)

        # Initialize the number of elephants detected
        num_elephants = 0

        # Check predictions from YOLO-NAS-M model
        if hasattr(predictions_nas_m, 'prediction'):
            pred_data = predictions_nas_m.prediction
            num_elephants += draw_bounding_boxes(filtered_frame, pred_data)

        # Check predictions from YOLO-V4 model
        # if hasattr(predictions_nas_m2, 'prediction'):
        #     pred_data = predictions_nas_m2.prediction
        #     num_elephants += draw_bounding_boxes(filtered_frame, pred_data)
        #
        # # Check predictions from YOLO-NAS-M Premium model
        # if hasattr(predictions_nas_m_premium, 'prediction'):
        #     pred_data = predictions_nas_m_premium.prediction
        #     num_elephants += draw_bounding_boxes(filtered_frame, pred_data)

        # Display the number of elephants detected
        cv2.putText(filtered_frame, f'Elephants Detected: {num_elephants}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display whether it is day or night
        day_or_night = is_day_or_night(frame)
        cv2.putText(filtered_frame, f'Time of Capture: {get_current_time_in_delhi()} | Day/Night: {day_or_night}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame with bounding boxes around detected elephants
        cv2.imshow('frame', filtered_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Switch to the morning model if it's morning
        if day_or_night == "Day":
            cap.release()
            cv2.destroyAllWindows()
            detect_yolo_nas_m()
            break

    # Release the camera resource when done
    cap.release()
    cv2.destroyAllWindows()

def is_day_or_night(frame, threshold=100):
    """
    Determine if the frame was captured during day or night.

    Args:
    - frame: The input frame from the webcam.
    - threshold: The brightness threshold to distinguish between day and night.

    Returns:
    - str: "Day" if it is considered day, "Night" if it is considered night.
    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the average brightness of the frame
    avg_brightness = np.mean(gray_frame)

    # Determine if it is day or night based on the threshold
    if avg_brightness > threshold:
        return "Day"
    else:
        return "Night"


def get_current_time_in_delhi():
    """
    Get the current time in Delhi, India.

    Returns:
    - str: The current time formatted as 'HH:MM AM/PM'.
    """
    delhi_tz = pytz.timezone('Asia/Kolkata')
    delhi_time = datetime.now(delhi_tz)
    return delhi_time.strftime('%I:%M %p')

# Function to perform real-time detection from video using YOLO model
yolo_nas_m2 = models.get("yolo_nas_m", pretrained_weights='coco')


# Define the class index for elephant (class index 20 in COCO dataset)

# Define the confidence threshold
confidence_threshold2 = 0.55

# Function to perform real-time detection from webcam using YOLO-NAS-M model
def detect_yolo_nas_m():
    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Perform prediction on the frame
        predictions = yolo_nas_m2.predict(frame)

        # Initialize the number of elephants detected
        num_elephants = 0

        # Check if predictions have 'prediction' attribute
        if hasattr(predictions, 'prediction'):
            # Extract the predictions
            pred_data = predictions.prediction

            # Process the prediction data
            for i in range(len(pred_data.labels)):
                label = pred_data.labels[i]
                confidence = pred_data.confidence[i]

                if label == elephant_class_index2 and confidence > confidence_threshold2:
                    num_elephants += 1
                    bbox = pred_data.bboxes_xyxy[i]  # Assuming the bounding boxes are stored in bboxes_xyxy
                    # Draw bounding box and label for the elephant
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Elephant {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the number of elephants detected
            cv2.putText(frame, f'Elephants Detected: {num_elephants}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with bounding boxes around detected elephants
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Switch to the night model if it's night
        day_or_night = is_day_or_night(frame)
        if day_or_night == "Night":
            cap.release()
            cv2.destroyAllWindows()
            detect_with_multiple_models()
            break

    # Release the camera resource when done
    cap.release()
    cv2.destroyAllWindows()


# Main function to handle model switching based on time of day
def main():
    while True:
        # Open webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Failed to capture frame from webcam.")
            break

        day_or_night = is_day_or_night(frame)

        if day_or_night == "Day":
            detect_yolo_nas_m()
        else:
            detect_with_multiple_models()

if __name__ == "__main__":
    main()
