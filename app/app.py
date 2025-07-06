from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model (adjust path if needed)
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'exam_score_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        previous_score = float(request.form['previous_score'])
        study_hours = float(request.form['study_hours'])
        class_participation = float(request.form['class_participation'])
        
        features = np.array([[previous_score, study_hours, class_participation]])
        predicted_score = model.predict(features)[0]
        
        return render_template('index.html', prediction_text=f'Estimated Exam Score: {predicted_score:.2f}')
    except Exception as e:
        return f"❌ Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
