import json
import numpy as np
import tensorflow as tf
from flask import Flask, request


#================Membuat Web App Sederhana Menggunakan Flask========================
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"
#===================================================================================


#============Membuat API Endpoint dalam Web App sebagai Model Serving===============
MODEL_PATH = "fashion-mnist"
model = tf.keras.models.load_model(MODEL_PATH)

def data_preprocessing(image):
    image = np.array(image) / 255.0
    image = (np.expand_dims(image, 0))
    return image

@app.route("/predict", methods=["POST"])
def predict():
    request_json = request.json
    image = data_preprocessing(request_json.get("data"))

    prediction = model.predict(image)
    prediction = tf.argmax(prediction[0]).numpy()

    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress',
        'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    response_json = {
        "prediction": class_names[prediction]
    }

    return json.dumps(response_json)
#===================================================================================



#=============================Menjalankan Web App===================================
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
#===================================================================================