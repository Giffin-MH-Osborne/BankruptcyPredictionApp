from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import csv
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__, template_folder='pages')
model = load_model('models/Final_RF_Model')

cols = []

with open('cols.csv', 'r') as f:
    lines = f.read().replace('\ufeff','')
    cols = lines.split(',')
    

@app.route('/')
def home():
    return render_template('index.html', pred='', cols=cols)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns= cols)
    prediction=predict_model(model, data=data_unseen, round=0)
    label = int(prediction.Label[0])
    result = ""
    if(bool(label)):
        result="go Bankrupt"
    else:
        result="not go Bankrupt"

    prediction=int(prediction.Label[0])
    return render_template('index.html', pred='The model predicts this company will {}'.format(result), cols=cols)