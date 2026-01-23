from cnnClassifier.components.multiobject_detection import MultiObjectDetector
import os

class MultiObjectPredictionPipeline:
    def __init__(self, filename, model_size='n'):
        """
        Args:
            filename: Input image path
            model_size: 'n' (nano/fast), 's' (small), 'm' (medium)
        """
        self.filename = filename
        self.detector = MultiObjectDetector(model_size=model_size)
    
    def predict(self, confidence_threshold=0.25):
        """Detect multiple objects in the image"""
        if not os.path.exists(self.filename):
            return {"error": "Image file not found"}
        
        try:
            # Detect objects and create annotated image
            output_path = "output_detected.jpg"
            result = self.detector.detect_and_draw(
                self.filename, 
                output_path, 
                confidence_threshold
            )
            
            return result
        except Exception as e:
            return {"error": str(e)}