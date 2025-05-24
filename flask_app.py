from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
with open('Model.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        input_data = request.json
        
        # Convert input data to a DataFrame
        user_input = pd.DataFrame([input_data])
        
        # Make predictions
        prediction = loaded_pipeline.predict(user_input)
        
        # Map prediction to human-readable label
        label_mapping = {0: "No Accident", 1: "Accident"}
        predicted_label = label_mapping[prediction[0]]
        
        # Get prediction probabilities
        prediction_proba = loaded_pipeline.predict_proba(user_input)
        confidence = max(prediction_proba[0]) * 100
        
        # Return JSON response
        response = {
            "prediction": predicted_label,
            "confidence": f"{confidence:.2f}%"
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
