import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
with open('Model.pkl', 'rb') as f:
    model = pickle.load(f)

# Default values for features
DEFAULT_VALUES = {
    'Weather': 'Clear',
    'Age': 5,
    'Type_of_Vehicle': 'Car',
    'Road_Type': 'City Road',
    'Time_of_Day': 'Night',
    'Traffic_Density': 1.0,
    'Speed_Limit': 50,
    'Number_of_Vehicles': 1,
    'Driver_Alcohol': 0.0,
    'Accident_Severity': 'Low',
    'Road_Condition': 'Dry',
    'Vehicle_Type': 'Car',
    'Driver_Age': 30,
    'Driver_Experience': 5,
    'Road_Light_Condition': 'Daylight',
    'Hour': 12,
    'Day': 15,
    'Month': 6,
    'DayOfWeek': 2  # Tuesday
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        
        # Create input dictionary with defaults
        input_data = DEFAULT_VALUES.copy()
        
        # Update with provided latitude and longitude
        input_data['Latitude'] = data['Latitude']
        input_data['Longitude'] = data['Longitude']
        
        # Optionally update with any other provided features
        for key in data:
            if key in input_data:
                input_data[key] = data[key]
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Prepare minimal response
        result = {
            'prediction': 'Accident' if prediction == 1 else 'No Accident',
            'confidence': round(float(max(probability)) * 100, 2)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)