from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import os

# Create output directory if it doesn't exist
output_dir = Path(r'C:\test_pdfs\test\evaluation_results')
output_dir.mkdir(exist_ok=True, parents=True)

# Load the model
model_path = 'best.pt'
model = YOLO(model_path)

# Get class names
data_dir = r'C:\test_pdfs\test'
with open(Path(data_dir) / 'obj.names', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Get image paths
with open(Path(data_dir) / 'train.txt', 'r') as f:
    img_paths = []
    for path in f.readlines():
        path = path.strip()
        if not (path.startswith('/') or (len(path) > 1 and path[1] == ':')):
            if path.startswith('data/'):
                path = path[5:]  # Remove 'data/' prefix
            full_path = str(Path(data_dir) / path)
            img_paths.append(full_path)
        else:
            img_paths.append(path)

# Process each image
for img_path in img_paths:
    # Get image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        continue
    
    # Get predictions with different confidence thresholds
    for conf_threshold in [0.5]:
        # Run prediction
        results = model.predict(img_path, conf=conf_threshold)
        predictions = results[0]
        
        # Create a copy of the image
        img_annotated = img.copy()
        
        # Draw prediction boxes (in blue)
        if len(predictions.boxes) > 0:
            boxes = predictions.boxes.xyxy.cpu().numpy()
            scores = predictions.boxes.conf.cpu().numpy()
            classes = predictions.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, cls_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box.astype(int)
                
                # Draw box
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw label
                cls_name = class_names[cls_id]
                cv2.putText(img_annotated, f"{cls_name} {score:.2f}", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Save the annotated image
        img_name = os.path.basename(img_path)
        output_path = output_dir / f"{os.path.splitext(img_name)[0]}_conf_{conf_threshold:.2f}.jpg"
        cv2.imwrite(str(output_path), img_annotated)
        
        print(f"Saved annotated image with confidence threshold {conf_threshold} to {output_path}")

print("Finished processing all images")
