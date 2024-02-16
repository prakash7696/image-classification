from flask import Flask, request, jsonify
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import io

app = Flask(__name__)


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

@app.route('/predict', methods=['POST'])
def predict():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    
    file = request.files['file']
    
   
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    return jsonify({'predicted_image': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
