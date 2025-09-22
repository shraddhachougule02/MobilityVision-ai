from ultralytics import YOLO
import cv2
import os
from collections import Counter

# -----------------------------
# CONFIGURATION
# -----------------------------
people_model_path = "../best2.pt"  # Path to your trained people YOLO model
vehicles_model_path = "../best.pt"  # Path to your trained vehicles YOLO model

input_image = (r"assests\11.png") 
output_folder = "../results/images"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
model_people = YOLO(people_model_path)
model_vehicles = YOLO(vehicles_model_path)

# -----------------------------
# RUN INFERENCE ON BOTH MODELS
# -----------------------------
results_people = model_people(input_image ,conf=0.30)
results_vehicles = model_vehicles(input_image , conf=0.25)

# -----------------------------
# PLOT AND COMBINE ANNOTATED IMAGES
# -----------------------------
annotated_people = results_people[0].plot()
annotated_vehicles = results_vehicles[0].plot()

# Combine both annotations on the same image by overlaying
combined_annotated = cv2.addWeighted(annotated_people, 0.5, annotated_vehicles, 0.5, 0)

# Save combined annotated image
image_name = os.path.basename(input_image)
output_path = os.path.join(output_folder, image_name)
cv2.imwrite(output_path, combined_annotated)

print(f"\n✅ Detection complete! Annotated image saved at: {output_path}\n")

# -----------------------------
# COUNT OBJECTS BY CLASS ACROSS BOTH MODELS
# -----------------------------
# Get classes and counts from people model
class_ids_people = [int(box.cls[0]) for box in results_people[0].boxes]
class_counts_people = Counter(class_ids_people)

# Get classes and counts from vehicles model
class_ids_vehicles = [int(box.cls[0]) for box in results_vehicles[0].boxes]
class_counts_vehicles = Counter(class_ids_vehicles)

# Map class ids to names
people_class_names = model_people.names
vehicles_class_names = model_vehicles.names

print("Detected counts for people model:")
for class_id, count in class_counts_people.items():
    print(f"- {people_class_names[class_id]}: {count}")

print("Detected counts for vehicles model:")
for class_id, count in class_counts_vehicles.items():
    print(f"- {vehicles_class_names[class_id]}: {count}")

# -----------------------------
# OPTIONAL: Print each detection (people)
# -----------------------------
print("\nPeople detections:")
for box in results_people[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = people_class_names[class_id]
    print(f"Detected: {class_name} with confidence {confidence:.2f}")

# -----------------------------
# OPTIONAL: Print each detection (vehicles)
# -----------------------------
print("\nVehicle detections:")
for box in results_vehicles[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = vehicles_class_names[class_id]
    print(f"Detected: {class_name} with confidence {confidence:.2f}")
