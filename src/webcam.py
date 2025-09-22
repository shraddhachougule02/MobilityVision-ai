from ultralytics import YOLO
import cv2

# Load separate YOLOv8 models for people and vehicles
model_people = YOLO("../best2.pt")
model_vehicles = YOLO("../best.pt")

# Open webcam (change index if multiple webcams)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open webcam")
    exit()

# Sets to keep track of unique IDs seen by each model
unique_people_ids = set()
unique_vehicle_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Track people with tracking + detection; lower conf threshold for vehicles
    results_people = model_people.track(frame, conf=0.25)
    results_vehicles = model_vehicles.track(frame, conf=0.25)

    # Plot annotated frames for overlay
    annotated_people = results_people[0].plot()
    annotated_vehicles = results_vehicles[0].plot()
    combined_annotated = cv2.addWeighted(annotated_people, 0.5, annotated_vehicles, 0.5, 0)

    # Extract unique IDs for people
    for box in results_people[0].boxes:
        if hasattr(box, "id") and box.id is not None:
            unique_people_ids.add(int(box.id))

    # Extract unique IDs for vehicles
    for box in results_vehicles[0].boxes:
        if hasattr(box, "id") and box.id is not None:
            unique_vehicle_ids.add(int(box.id))

    # Show the combined frame
    cv2.imshow("YOLOv8 People + Vehicles Tracking & Counting", combined_annotated)

    # Print real-time count
    print(f"Unique people detected so far: {len(unique_people_ids)}")
    print(f"Unique vehicles detected so far: {len(unique_vehicle_ids)}")

    # Exit if 'q' or Esc key is pressed
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()

# Print final counts
print("\nFinal unique people count:", len(unique_people_ids))
print("Final unique vehicles count:", len(unique_vehicle_ids))
