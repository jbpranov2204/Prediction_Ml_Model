from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def train_model():
    # Read and preprocess data
    data = pd.read_csv("data.csv")
    data['Accident'] = data['Accident'].replace(np.nan, 0)
    
    # Encode target variable
    data['Accident'] = LabelEncoder().fit_transform(data['Accident'])
    
    # Convert Timestamp and extract features
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Hour'] = data['Timestamp'].dt.hour
    data['Day'] = data['Timestamp'].dt.day
    data['Month'] = data['Timestamp'].dt.month
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek

    # Define features
    features = [
        'Latitude', 'Longitude', 'Weather', 'Age', 'Type_of_Vehicle', 'Road_Type', 
        'Time_of_Day', 'Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 
        'Driver_Alcohol', 'Accident_Severity', 'Road_Condition', 'Vehicle_Type', 
        'Driver_Age', 'Driver_Experience', 'Road_Light_Condition', 'Hour', 'Day', 
        'Month', 'DayOfWeek'
    ]
    
    X = data[features]
    y = data['Accident']

    # Define feature types
    categorical_features = [
        'Weather', 'Type_of_Vehicle', 'Road_Type', 'Time_of_Day', 'Traffic_Density',
        'Driver_Alcohol', 'Accident_Severity', 'Road_Condition', 'Vehicle_Type',
        'Road_Light_Condition'
    ]
    numerical_features = [
        'Latitude', 'Longitude', 'Age', 'Speed_Limit', 'Number_of_Vehicles',
        'Driver_Age', 'Driver_Experience', 'Hour', 'Day', 'Month', 'DayOfWeek'
    ]

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    # Create model pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42))
    ])

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model
    with open('model.pkl', 'wb') as file:
        pickle.dump(pipeline, file)

    return {
        'accuracy': float(accuracy),
        'report': classification_report(y_test, y_pred)
    }

@app.route('/model', methods=['POST'])
def handle_model():
    try:
        operation = request.args.get('operation', 'predict')
        
        if operation == 'train':
            results = train_model()
            return jsonify({
                'status': 'success',
                'message': 'Model trained successfully',
                'accuracy': results['accuracy'],
                'report': results['report']
            })
        elif operation == 'predict':
            # Load model
            with open('Model.pkl', 'rb') as file:
                model = pickle.load(file)

            # Get input data
            input_data = request.get_json()
            
            # Convert to DataFrame
            user_input = pd.DataFrame([input_data])

            # Make prediction
            prediction = model.predict(user_input)
            prediction_proba = model.predict_proba(user_input)

            # Format response
            label_mapping = {0: "No Accident", 1: "Accident"}
            confidence = float(max(prediction_proba[0]) * 100)

            return jsonify({
                'status': 'success',
                'prediction': label_mapping[prediction[0]],
                'confidence': confidence
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Invalid operation. Use "train" or "predict"'
            }), 400

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
