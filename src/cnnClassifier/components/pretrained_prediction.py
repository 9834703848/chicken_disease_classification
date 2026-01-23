import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

class PretrainedPredictor:
    def __init__(self):
        # Load pre-trained MobileNetV2 with ImageNet weights
        print("Loading pre-trained model...")
        self.model = MobileNetV2(weights='imagenet', include_top=True)
        print("Model loaded successfully!")
    
    def predict_image(self, image_path):
        """
        Predict what's in the image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of predictions with class names and probabilities
        """
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        predictions = self.model.predict(img_array)
        
        # Decode predictions to human-readable labels
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        
        # Format results
        results = []
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            results.append({
                "rank": i + 1,
                "label": label.replace('_', ' ').title(),
                "confidence": f"{score * 100:.2f}%"
            })
        
        return results