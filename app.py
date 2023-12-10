import os
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

# Import the disease dictionary from disease.py
from disease import disease_dic

model = load_model(r'C:\Users\switc\Desktop\SEM 7 PROJECT\Crop Disease Detection\Crop Disease Detection\Crop Leaves Disease Detection Training\best_model.h5')

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prediction(path):
    ref = {
        0: 'Apple___Apple_scab',
        1: 'Apple___Black_rot',
        2: 'Apple___Cedar_apple_rust',
        # Add more disease labels here...
    }

    img = load_img(path, target_size=(256, 256))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    pred = np.argmax(model.predict(img))

    pred_label = ref[pred]

    # Get disease information and solutions from disease_dic
    if pred_label in disease_dic:
        disease_info = disease_dic[pred_label]
    else:
        disease_info = "Disease information not available."

    return pred_label, disease_info

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        pred_label, disease_info = prediction(UPLOAD_FOLDER + '/' + filename)
        return render_template('predict.html', org_img_name=filename, prediction=pred_label, disease_info=disease_info)

if __name__ == '__main__':
    app.run(debug=True)
