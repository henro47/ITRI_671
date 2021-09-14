import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import sys

EPOCHS = 117
BATCH_SIZE = 224

def load_model(model_path):
    try:
        model = keras.load_model(model_path)
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