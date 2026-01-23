from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.pretrained_predict import PretrainedPredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = None  # Will be initialized on first prediction


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)
        
        # Initialize classifier on first use (lazy loading)
        if clApp.classifier is None:
            clApp.classifier = PretrainedPredictionPipeline(clApp.filename)
        
        result = clApp.classifier.predict()
        return jsonify(result)
    except Exception as e:
        return jsonify([{"error": str(e)}])


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=5000)