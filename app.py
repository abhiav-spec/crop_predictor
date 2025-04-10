import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('crop_model.pkl')


@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction=model.predict(final_features)
    return render_template('index.html', prediction=f"The predicted crop is {prediction[0]}")

if __name__ == "__main__":
    app.run(debug=True)   
