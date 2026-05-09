import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load YOLOv8n model (you can also try YOLOv8s if you have the compute)
# Load YOLOv8m model for better accuracy in detecting small details or occluded persons
model = YOLO('yolov8m.pt')

def apply_transformations(roi):
    """Applies a sequence of transformations to the Region of Interest (ROI)"""
    
    # 1. Edge Detection / Cartoonish Effect
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(roi, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # 2. Add a slight color tint (e.g., thermal-like or creepy green)
    # Let's add a green tint
    tint = np.zeros_like(cartoon)
    tint[:, :, 1] = 50 # Add green
    tinted = cv2.addWeighted(cartoon, 1, tint, 0.5, 0)
    
    # 3. Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(tinted, -1, kernel)

    return sharpened

def process_video(input_path, output_path):
    print(f"Processing {input_path}...")
    
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count}/{total_frames}")

        results = model(frame, conf=0.15, imgsz=1280, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Extract ROI
                roi = frame[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue

                # Apply transformations
                processed_roi = apply_transformations(roi)
                
                # Insert back into frame
                frame[y1:y2, x1:x2] = processed_roi
                
                # Get class name
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name.capitalize(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"Finished processing. Saved to {output_path}")

# List of videos to process
videos = [
    "horizon.mp4",
    "lotr.mp4",
    "need_for_speed.mp4",
    "uncharted.mp4"
]

input_dir = r"d:\Semester 4\AI LAB\OEL"
output_dir = os.path.join(input_dir, "task1_output")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for video_name in videos:
    input_path = os.path.join(input_dir, video_name)
    output_path = os.path.join(output_dir, "processed_" + video_name)
    
    # Check if the input file exists first
    if os.path.exists(input_path):
        process_video(input_path, output_path)
    else:
        # Check if there's a version with a different extension or name variation
        print(f"Warning: Could not find {input_path}")
