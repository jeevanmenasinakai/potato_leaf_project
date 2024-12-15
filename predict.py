import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("potato_model.keras")
# Function to preprocess the input image
def prepare(filepath):
    img = image.load_img(filepath, target_size=(128, 128))  # Resize to 128x128
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize image values
    return img_array
# Predict on a new image
def predict(filepath):
    img_array = prepare(filepath)  # Preprocess the image
    prediction = model.predict(img_array)  # Get model prediction
    class_names = ['Early Blight', 'Late Blight', 'Healthy']  # Your class names
    predicted_class = class_names[np.argmax(prediction)]  # Get the predicted class
    print(f"The leaf is classified as: {predicted_class}")
# Test the model with an image
predict("test_leaf.jpg")