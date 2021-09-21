from django.http import HttpResponse
from django.shortcuts import render
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import sys

# KERAS API FUNCTIONS

EPOCHS = 117
BATCH_SIZE = 224

def load_model(model_path):
    try:
        model = keras.models.load_model(model_path)
        return model
    except:
        print('failed to load model')
        sys.exit()

def load_prediction_data(data_path):
    try:
        predict_data = pd.read_csv(data_path)
        return predict_data
    except:
        print('Data import Failed')
        sys.exit()

def predict(model, data):
    predictions = model.predict(x=data, batch_size=BATCH_SIZE, verbose=0)
    return predictions

def get_category(index):
    img_path = ''
    if(index == 0):
        return 'Bricks'
    elif(index == 1):
        return 'Epoxy'
    elif(index == 2):
        return 'Grass'
    elif(index == 3):
        return 'Dirt'
    elif(index == 4):
        return 'Stone'
    elif(index == 5):
        return 'Tarmac'

#VIEWS

def home(request):
    return render(request, 'home.html')

def prediction(request):
    category = "CAT"
    if(request.method == 'POST'):
        uploaded_file = request.FILES['document']
        model = load_model('ITRI_671\keras_model')
        df = pd.read_csv(uploaded_file, delimiter=',', header=None)
        pred = predict(model, df)
        index = tf.argmax(pred, axis=1)
        category = get_category(index)
           
    return render(request, "prediction.html",{'results':category})

	