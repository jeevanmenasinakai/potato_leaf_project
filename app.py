from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model/potato_model.keras')

# Define image preprocessing function
def prepare(filepath):
    img = image.load_img(filepath, target_size=(128, 128))  # Resize to 128x128
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize image values
    return img_array

# Define prediction function
def predict(filepath):
    img_array = prepare(filepath)
    prediction = model.predict(img_array)
    class_names = ['Early Blight', 'Late Blight', 'Healthy']  # Your class names
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['file']
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Predict the class of the uploaded image
        result = predict(filepath)

        return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
import os
app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
