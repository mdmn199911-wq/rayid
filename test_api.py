
import requests
import json

# The URL of our Flask API
url = 'http://127.0.0.1:5000/predict'

# --- Test Case 1: Data for a likely diabetic patient ---
# High glucose and HbA1c levels
data_diabetic = {
    'HbA1c_level': 7.0,
    'blood_glucose_level': 180,
    'bmi': 31.5,
    'age': 55,
    'hypertension': 1,
    'heart_disease': 0
}

# --- Test Case 2: Data for a likely healthy patient ---
# Normal glucose and HbA1c levels
data_healthy = {
    'HbA1c_level': 5.5,
    'blood_glucose_level': 100,
    'bmi': 22.0,
    'age': 30,
    'hypertension': 0,
    'heart_disease': 0
}

def test_prediction(data, description):
    """Sends data to the API and prints the response."""
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(data), headers=headers)
        
        print(f"--- Testing: {description} ---")
        print(f"Input Data: {data}")
        
        if response.status_code == 200:
            print(f"Prediction Response: {response.json()}")
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response Content: {response.text}")
        print("-" * 30)

    except requests.exceptions.ConnectionError as e:
        print("--- CONNECTION ERROR ---")
        print("Could not connect to the server.")
        print("Please make sure the Flask app (`app.py`) is running in a separate terminal.")
        print("-" * 30)

if __name__ == '__main__':
    test_prediction(data_diabetic, "Likely Diabetic Patient")
    test_prediction(data_healthy, "Likely Healthy Patient")

