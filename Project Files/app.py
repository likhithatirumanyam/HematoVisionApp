# app.py
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("Blood_Cell.h5")

# Define class names (modify according to your model's output)
class_names = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    preds = model.predict(img_array)
    return class_names[np.argmax(preds)], np.max(preds)

@app.route('/')
def home():
    print("Home page loaded")
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    if file:
        filename = file.filename
        filepath = os.path.join("static", filename)
        file.save(filepath)
        prediction, confidence = model_predict(filepath)
        return render_template('result.html', 
                            prediction=prediction, 
                       confidence=round(confidence * 100, 2), 
                       image_filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
