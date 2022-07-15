import json
import tensorflow as tf
from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"

# Load model
MODEL_PATH = "fashion-mnist"
model = tf.keras.models.load_model(MODEL_PATH)

# Make prediction route
@app.route("/predict", methods=["POST"])
def predict():
    request_json = request.json

    prediction = model.predict(request_json.get("data"))
    prediction = tf.argmax(prediction[0]).numpy()

    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress',
        'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    response_json = {
        "prediction": class_names[prediction]
    }

    return json.dumps(response_json)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
