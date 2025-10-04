#!/usr/bin/env python3
"""
Machine Learning Predictor for Alzheimer's Detection
This script handles model loading and prediction logic
"""

import sys
import json
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AlzheimerPredictor:
    def __init__(self):
        """Initialize the predictor with trained models"""
        try:
            # Load the trained models
            self.voting_model = joblib.load("alzheimers_voting_model.pkl")
            self.stacking_model = joblib.load("alzheimers_stacking_model.pkl")
            print("Models loaded successfully", file=sys.stderr)
        except Exception as e:
            print(f"Error loading models: {e}", file=sys.stderr)
            sys.exit(1)
    
    def preprocess_input(self, data):
        """Preprocess input data to match training format"""
        try:
            # Extract values from input data
            age = float(data.get('age', 60))
            gender = data.get('gender', 'Male')
            ethnicity = data.get('ethnicity', 'Caucasian')
            education = data.get('education', 'Bachelor\'s')
            bmi = float(data.get('bmi', 22.0))
            smoking = data.get('smoking', 'No')
            alcohol = float(data.get('alcohol', 2.0))
            physical_activity = float(data.get('physical_activity', 5.0))
            diet_quality = float(data.get('diet_quality', 6.0))
            sleep_quality = float(data.get('sleep_quality', 5.0))
            family_history = data.get('family_history', 'No')
            cardiovascular = data.get('cardiovascular', 'No')
            diabetes = data.get('diabetes', 'No')
            depression = data.get('depression', 'No')
            head_injury = data.get('head_injury', 'No')
            hypertension = data.get('hypertension', 'No')
            systolic_bp = float(data.get('systolic_bp', 120))
            diastolic_bp = float(data.get('diastolic_bp', 80))
            cholesterol_total = float(data.get('cholesterol_total', 200.0))
            cholesterol_ldl = float(data.get('cholesterol_ldl', 120.0))
            cholesterol_hdl = float(data.get('cholesterol_hdl', 50.0))
            cholesterol_triglycerides = float(data.get('cholesterol_triglycerides', 150.0))
            mmse = float(data.get('mmse', 25))
            functional_assessment = float(data.get('functional_assessment', 7.0))
            memory_complaints = data.get('memory_complaints', 'No')
            behavioral_problems = data.get('behavioral_problems', 'No')
            adl = float(data.get('adl', 8.0))
            confusion = data.get('confusion', 'No')
            disorientation = data.get('disorientation', 'No')
            personality_changes = data.get('personality_changes', 'No')
            difficulty_tasks = data.get('difficulty_tasks', 'No')
            
            # Convert categorical variables to binary
            gender_val = 1 if gender == 'Male' else 0
            smoking_val = 1 if smoking == 'Yes' else 0
            family_val = 1 if family_history == 'Yes' else 0
            cardiovascular_val = 1 if cardiovascular == 'Yes' else 0
            diabetes_val = 1 if diabetes == 'Yes' else 0
            depression_val = 1 if depression == 'Yes' else 0
            head_injury_val = 1 if head_injury == 'Yes' else 0
            hypertension_val = 1 if hypertension == 'Yes' else 0
            memory_val = 1 if memory_complaints == 'Yes' else 0
            behavioral_val = 1 if behavioral_problems == 'Yes' else 0
            confusion_val = 1 if confusion == 'Yes' else 0
            disorientation_val = 1 if disorientation == 'Yes' else 0
            personality_val = 1 if personality_changes == 'Yes' else 0
            difficulty_val = 1 if difficulty_tasks == 'Yes' else 0
            
            # Create base features (31 features)
            base_features = [
                age, gender_val, bmi,
                smoking_val, alcohol, physical_activity, diet_quality, sleep_quality,
                family_val, cardiovascular_val, diabetes_val, depression_val, head_injury_val,
                hypertension_val, systolic_bp, diastolic_bp, cholesterol_total, cholesterol_ldl,
                cholesterol_hdl, cholesterol_triglycerides, mmse, functional_assessment,
                memory_val, behavioral_val, adl, confusion_val, disorientation_val,
                personality_val, difficulty_val
            ]
            
            # Add engineered features (3 more features)
            age_bmi = age * bmi
            pa_diet = physical_activity * diet_quality
            chol_ratio = cholesterol_total / (cholesterol_hdl + 1e-5)
            
            base_features.extend([age_bmi, pa_diet, chol_ratio])
            
            # Ethnicity one-hot encoding
            ethnicity_map = {
                'Caucasian': [0, 0, 0, 0],
                'African American': [1, 0, 0, 0],
                'Hispanic': [0, 1, 0, 0],
                'Asian': [0, 0, 1, 0],
                'Other': [0, 0, 0, 1]
            }
            ethnicity_features = ethnicity_map.get(ethnicity, [0, 0, 0, 0])
            
            # Education one-hot encoding
            education_map = {
                'High School': [0, 0, 0],
                'Bachelor\'s': [1, 0, 0],
                'Master\'s': [0, 1, 0],
                'PhD': [0, 0, 1]
            }
            education_features = education_map.get(education, [0, 0, 0])
            
            # Combine all features and take exactly 39 features
            all_features = base_features + ethnicity_features + education_features
            features = all_features[:39]  # Ensure exactly 39 features
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error in preprocessing: {e}", file=sys.stderr)
            return None
    
    def predict(self, data):
        """Make predictions using both models"""
        try:
            # Preprocess input
            input_features = self.preprocess_input(data)
            if input_features is None:
                return None
            
            # Get predictions
            voting_pred = self.voting_model.predict(input_features)[0]
            stacking_pred = self.stacking_model.predict(input_features)[0]
            
            # Get prediction probabilities
            voting_prob = self.voting_model.predict_proba(input_features)[0]
            stacking_prob = self.stacking_model.predict_proba(input_features)[0]
            
            return {
                'voting_prediction': int(voting_pred),
                'stacking_prediction': int(stacking_pred),
                'voting_probabilities': {
                    'low_risk': float(voting_prob[0]),
                    'high_risk': float(voting_prob[1])
                },
                'stacking_probabilities': {
                    'low_risk': float(stacking_prob[0]),
                    'high_risk': float(stacking_prob[1])
                },
                'models_agree': bool(voting_pred == stacking_pred),
                'feature_count': int(input_features.shape[1])
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}", file=sys.stderr)
            return None

def main():
    """Main function to handle command line input"""
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        
        # Initialize predictor
        predictor = AlzheimerPredictor()
        
        # Make prediction
        result = predictor.predict(input_data)
        
        if result:
            print(json.dumps(result))
        else:
            print(json.dumps({'error': 'Prediction failed'}))
            
    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)

if __name__ == "__main__":
    main()
