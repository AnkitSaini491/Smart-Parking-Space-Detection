
import cv2
from ultralytics import YOLO
from collections import Counter

# Load YOLO model
model = YOLO("yolov8n.pt")

# Vehicle classes
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


def detect_vehicles(video_path):

    cap = cv2.VideoCapture(video_path)

    vehicle_count = Counter()

    if not cap.isOpened():
        print("Error: Cannot open video")
        return vehicle_count

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # YOLO detection
        results = model(frame, verbose=False)

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id in vehicle_classes and confidence > 0.5:

                    vehicle_name = vehicle_classes[class_id]

                    vehicle_count[vehicle_name] += 1

        # Display detection
        annotated_frame = results[0].plot()

        cv2.imshow(
            "AI Traffic Monitoring",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return vehicle_count


# Main program
if __name__ == "__main__":

    video_path = "dataset/traffic_video.mp4"

    counts = detect_vehicles(video_path)

    print("\nTRAFFIC DETECTION REPORT")
    print("------------------------")

    for vehicle, count in counts.items():
        print(vehicle, ":", count)

    print("\nTotal Vehicle Detections:", sum(counts.values()))

