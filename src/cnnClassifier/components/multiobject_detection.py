from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

class MultiObjectDetector:
    def __init__(self, model_size='n'):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
                       'n' is fastest but least accurate, 'x' is most accurate but slowest
        """
        print(f"Loading YOLOv8{model_size} model...")
        # This will auto-download the model on first run
        self.model = YOLO(f'yolov8{model_size}.pt')
        print("Model loaded successfully!")
    
    def detect_objects(self, image_path, confidence_threshold=0.25):
        """
        Detect multiple objects in an image
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence for detections (0.0 to 1.0)
            
        Returns:
            List of detected objects with their details
        """
        # Run inference
        results = self.model(image_path, conf=confidence_threshold)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detections.append({
                    "object": class_name.replace('_', ' ').title(),
                    "confidence": f"{confidence * 100:.2f}%",
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }
                })
        
        return detections
    
    def detect_and_draw(self, image_path, output_path="output_detected.jpg", confidence_threshold=0.25):
        """
        Detect objects and save image with bounding boxes
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detections and path to annotated image
        """
        # Run inference
        results = self.model(image_path, conf=confidence_threshold)
        
        # Plot results
        annotated_img = results[0].plot()
        
        # Save annotated image
        cv2.imwrite(output_path, annotated_img)
        
        # Get detections
        detections = self.detect_objects(image_path, confidence_threshold)
        
        return {
            "detections": detections,
            "annotated_image_path": output_path,
            "total_objects": len(detections)
        }