# Import all project dependencies
import os
import requests
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import cv2
from flask import Flask, request, jsonify

# Load the pretrained model
with open('fashion_model_flask.json', 'r') as f:
    json_model = f.read()

f_model = tf.keras.models.model_from_json(json_model)

# load weights into new model
f_model.load_weights('fashion_model_flask.h5')

print(f_model.input_shape)

# Creating the Flask API
## Defining the Flask application
app = Flask(__name__)

## Defining the classifyImage function
@app.route('/api/v1/<string:img_name>', methods=['POST'])
def classifyImage(img_name):
    upload_dir = 'Uploads/'
    image = cv2.imread(upload_dir+img_name, 0)
    print(image.shape)
    image = cv2.resize(image, (28, 28))
    # print(image.shape)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #(28, 28)
    # image = cv2.bitwise_not(image)
    print(image.shape)
    
    image = image/255
    image = image.reshape(1, 28*28)
    print(image.shape)
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    prediction = f_model.predict([image])
    
    return jsonify({'object_detected': classes[np.argmax(prediction[0])]})

# Starting the Flask application
app.run(port=5000, debug=False)