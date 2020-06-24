import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
clf_rf = pickle.load(open('Iris_Flower_Predictor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = clf_rf.predict(final_features)

    return render_template('contact.html', prediction_text='The Iris Flower Species is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)