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
        model = keras.models.load_model('ITRI_671\keras_model')
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


#VIEWS

def home(request):
    return render(request, 'home.html')

def prediction(request):
    model = load_model('keras_model')
    model.summary()
    input_data = []

    input_data.append(float(request.GET.get('t_h',1)))
    input_data.append(float(request.GET.get('t_p',1)))
    input_data.append(float(request.GET.get('pr',1)))
    input_data.append(float(request.GET.get('hu',1)))
    input_data.append(float(request.GET.get('yw',1)))
    input_data.append(float(request.GET.get('pi',1)))
    input_data.append(float(request.GET.get('ro',1)))
    input_data.append(float(request.GET.get('m_x',1)))
    input_data.append(float(request.GET.get('m_y',1)))
    input_data.append(float(request.GET.get('m_z',1)))
    input_data.append(float(request.GET.get('acc_x',1)))
    input_data.append(float(request.GET.get('acc_y',1)))
    input_data.append(float(request.GET.get('acc_z',1)))
    input_data.append(float(request.GET.get('gy_x',1)))
    input_data.append(float(request.GET.get('gy_y',1)))
    input_data.append(float(request.GET.get('gy_z',1)))

    df = pd.DataFrame([input_data])
    print(df.head())
    pred = predict(model, df)
    print(pred)
    return render(request, "prediction.html")

	