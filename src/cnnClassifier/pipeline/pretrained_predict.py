from cnnClassifier.components.pretrained_prediction import PretrainedPredictor
import os

class PretrainedPredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.predictor = PretrainedPredictor()
    
    def predict(self):
        """Make prediction on the uploaded image"""
        if not os.path.exists(self.filename):
            return [{"error": "Image file not found"}]
        
        try:
            results = self.predictor.predict_image(self.filename)
            return results
        except Exception as e:
            return [{"error": str(e)}]