from flask import Flask, render_template, request
import pandas as pd
from pycaret.regression import load_model, predict_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    model = load_model('D:\MuraKon\pycaret-deployment\predictor')

    # Get data from the form
    age = request.form['age']
    sex = request.form['sex']
    bmi = request.form['bmi']
    children = request.form['children']
    smoker = request.form['smoker']
    region = request.form['region']

    # Prepare unseen data
    data_unseen = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    # Make a prediction
    prediction = predict_model(model, data=data_unseen)
    
    # Return the prediction to the user
    return str(prediction.iloc[0]['prediction_label'])  # Assuming the label is the first column in the prediction

if __name__ == '__main__':
    app.run(debug=True)
