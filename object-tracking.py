import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import time
import argparse

# Set up command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="yolov10m.pt", help="YOLO model path")
parser.add_argument("--video", type=str, default="inference/cow.mp4", help="Path to input video or webcam index (0)")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence Threshold for detection")
args = parser.parse_args()

# Function to display FPS on the frame
def show_fps(frame, fps):
    x, y, w, h = 10, 10, 350, 50
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)  # Draw a rectangle for the FPS text background
    cv2.putText(frame, "FPS: " + str(fps), (20, 52), cv2.FONT_HERSHEY_PLAIN, 3.5, (0, 255, 0), 3)  # Add FPS text

if __name__ == '__main__':
    # Set up video capture
    video_input = args.video
    if video_input.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)  # Open webcam if video_input is a digit
    else:
        cap = cv2.VideoCapture(video_input)  # Open video file

    conf_thres = args.conf  # Confidence threshold for detection

    model = YOLO(args.model)  # Load YOLOv10 model

    track_history = defaultdict(lambda: [])  # Track history of detected objects
    start_time = 0  # Initialize start time for FPS calculation

    while cap.isOpened():
        success, frame = cap.read()  # Read a frame from the video
        annotated_frame = frame

        if success:
            # Perform object tracking using YOLOv10
            results = model.track(frame, classes=19, persist=True, tracker="bytetrack.yaml", conf=conf_thres)

            boxes = results[0].boxes.xywh.cpu()  # Get bounding boxes

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()  # Get track IDs
                class_ids = results[0].boxes.cls.int().cpu().tolist()  # Get class IDs

                # Plot the results on the frame
                annotated_frame = results[0].plot()

                # Draw tracking lines for each object
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # Append center point of the bounding box
                    if len(track) > 90:  # Retain track history for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(255, 0, 0),
                        thickness=3,
                    )

            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            
            start_time = end_time

            # Show FPS on the frame
            fps = float("{:.2f}".format(fps))
            show_fps(annotated_frame, fps)

            # Display the annotated frame
            cv2.imshow("YOLOv10 Cow Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
