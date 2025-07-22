from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your classifier model
model = joblib.load('tuned_gradient_boosting_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([np.array(features)])
        result = prediction[0]
        return render_template('index.html', prediction_text=f"Predicted Income: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
