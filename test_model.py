import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("models/plant_disease_cnn.h5")

# Class labels (same order as training)
class_labels = [
    'Tomato_Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Tomato_Late_blight',
    'Tomato_healthy',
    'Tomato_Septoria_leaf_spot',
    'Potato___Late_blight',
    'Tomato_Early_blight'
]

def predict_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    print(f"Prediction: {class_labels[class_idx]} ({confidence*100:.2f}%)")

if __name__ == "__main__":
    test_img = "processed_dataset/test/Pepper__bell___healthy/0e69c47d-72c6-4fc6-9437-910c95b183dc___JR_HL 8113.JPG"  # <-- put any leaf image here
    predict_image(test_img)
