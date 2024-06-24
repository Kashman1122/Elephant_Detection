import cv2
from super_gradients.training import models

# Load your custom YOLO-NAS-M model
yolo_nas_m = models.get('yolo_nas_m', num_classes=1, checkpoint_path="yolo-nas-m-premium.pth")

# Define the class index for elephant (class index 0 in custom model)
elephant_class_index = 0

# Define the confidence threshold
confidence_threshold = 0.75

# Define the maximum allowed box dimensions (as a fraction of the frame size)
max_box_fraction = 0.85  # Adjust as needed to ignore too large boxes

# Function to perform real-time detection from webcam using YOLO-NAS-M model
def detect_with_yolo_nas_m():
    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Perform prediction on the frame
        predictions = yolo_nas_m.predict(frame)

        # Initialize the number of elephants detected
        num_elephants = 0

        # Get the frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Check if predictions have 'prediction' attribute
        if hasattr(predictions, 'prediction'):
            # Extract the predictions
            pred_data = predictions.prediction

            # Process the prediction data
            for i in range(len(pred_data.labels)):
                label = pred_data.labels[i]
                confidence = pred_data.confidence[i]

                if label == elephant_class_index and confidence > confidence_threshold:
                    bbox = pred_data.bboxes_xyxy[i]  # Assuming the bounding boxes are stored in bboxes_xyxy
                    x1, y1, x2, y2 = map(int, bbox)

                    # Calculate box dimensions
                    box_width = x2 - x1
                    box_height = y2 - y1

                    # Check if the box is too large
                    if box_width > max_box_fraction * frame_width or box_height > max_box_fraction * frame_height:
                        continue

                    num_elephants += 1

                    # Draw bounding box and label for the elephant
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Elephant {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the number of elephants detected
            cv2.putText(frame, f'Elephants Detected: {num_elephants}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with bounding boxes around detected elephants
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera resource when done
    cap.release()
    cv2.destroyAllWindows()

# Main function to handle user input
def main():
    while True:
        user_input = input("Enter 'm' to run YOLO-NAS-M model, or 'q' to quit: ")
        if user_input == 'm':
            detect_with_yolo_nas_m()
        elif user_input == 'q':
            break
        else:
            print("Invalid input. Please enter 'm' or 'q'.")

if __name__ == "__main__":
    main()
