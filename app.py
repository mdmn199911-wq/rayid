
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS # Import CORS
import joblib
import os

# Initialize the Flask application
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Load Model and Scaler at startup ---
# This is more efficient than loading them for each request
models_dir = "trained_models"
scaler_path = os.path.join(models_dir, 'scaler.joblib')
model_path = os.path.join(models_dir, 'neural_network_model.joblib')

try:
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    print("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please make sure the 'trained_models' directory and its contents exist.")
    model = None # Set model to None if loading fails

# Serve the frontend files so the app can be hosted together with the API
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    allowed_files = {'index.html', 'script.js', 'style.css'}
    if filename in allowed_files and os.path.isfile(filename):
        return send_from_directory('.', filename)
    abort(404)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        
        # The order of features must be the same as during training
        features = [
            data['HbA1c_level'],
            data['blood_glucose_level'],
            data['bmi'],
            data['age'],
            data['hypertension'],
            data['heart_disease']
        ]

        # Convert to numpy array and scale
        final_features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Get the prediction result (0 or 1)
        output = int(prediction[0])

        # Provide a human-readable diagnosis
        diagnosis = 'Diabetic' if output == 1 else 'Not Diabetic'

        # Get risk level and recommendations
        risk_level, recommendations = get_risk_and_recommendations(data, output)

        # Return the full result as JSON
        return jsonify({
            'prediction': output,
            'diagnosis': diagnosis,
            'risk_level': risk_level,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_risk_and_recommendations(data, prediction):
    recommendations = []
    risk_level = "Low"

    # Determine Risk Level
    if prediction == 1:
        risk_level = "High"
    elif data['HbA1c_level'] >= 6.0 or data['blood_glucose_level'] > 140:
        risk_level = "Medium"
    elif data['bmi'] >= 25:
        risk_level = "Medium"

    # Generate Recommendations
    if prediction == 1:
        recommendations.append("Consult a healthcare professional immediately for a formal diagnosis and treatment plan.")
        recommendations.append("It is crucial to monitor your blood sugar levels regularly as advised by your doctor.")
    else:
        recommendations.append("Your result is negative, which is great news. Maintain a healthy lifestyle to keep it that way.")

    if data['bmi'] >= 30:
        recommendations.append("Your BMI indicates obesity. It is highly recommended to consult a doctor or nutritionist to create a weight management plan.")
    elif data['bmi'] >= 25:
        recommendations.append("Your BMI is in the overweight range. Incorporating regular physical activity and a balanced diet is recommended.")

    if data['hypertension'] == 1:
        recommendations.append("Continue to manage your hypertension and follow your doctor's advice.")

    if data['heart_disease'] == 1:
        recommendations.append("Given your history of heart disease, a healthy lifestyle is especially important. Continue to follow your cardiologist's recommendations.")

    if not recommendations:
        recommendations.append("Maintain a balanced diet and regular exercise. Regular check-ups are always a good practice.")

    return risk_level, recommendations


# Main entry point
if __name__ == '__main__':
    # To run this app: use the command `flask run` in your terminal
    # or simply `python app.py` and it will be accessible at http://127.0.0.1:5000
    print("Starting Flask server...")
    print("Access the prediction endpoint at http://127.0.0.1:5000/predict via POST request.")
    app.run(debug=True)
