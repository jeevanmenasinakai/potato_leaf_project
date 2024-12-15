from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model/potato_model.keras')

# Create the uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    class_names = ['Early Blight', 'Late Blight', 'Healthy']  # Update as per your model classes
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save the file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Predict the class of the uploaded image
    result = predict(filepath)

    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT or default to 5000
    app.run(host="0.0.0.0", port=port)
