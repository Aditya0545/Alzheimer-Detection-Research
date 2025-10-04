#!/usr/bin/env python3
"""
Enhanced Model Training Script for Alzheimer's Detection
This script includes advanced feature engineering, hyperparameter tuning, and model optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, roc_auc_score, roc_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, StackingClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedAlzheimerModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the dataset with enhanced feature engineering"""
        logging.info("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Remove unnecessary columns
        df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)
        
        # Handle categorical variables
        categorical_cols = ['Gender', 'Ethnicity', 'EducationLevel']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
        
        # Separate features and target
        X = df_encoded.drop('Diagnosis', axis=1)
        y = df_encoded['Diagnosis']
        
        # Advanced feature engineering
        X = self._engineer_features(X)
        
        # Feature selection based on importance
        X = self._select_features(X, y)
        
        return X, y
    
    def _engineer_features(self, X):
        """Advanced feature engineering"""
        logging.info("Engineering new features...")
        
        # Interaction features
        X['Age_BMI'] = X['Age'] * X['BMI']
        X['PA_Diet'] = X['PhysicalActivity'] * X['DietQuality']
        X['Sleep_Diet'] = X['SleepQuality'] * X['DietQuality']
        
        # Health risk scores
        X['Cardiovascular_Risk'] = (
            X['CardiovascularDisease'] + X['Hypertension'] + 
            X['Diabetes'] + (X['SystolicBP'] > 140).astype(int)
        )
        
        X['Cognitive_Symptoms'] = (
            X['MemoryComplaints'] + X['Confusion'] + X['Disorientation'] +
            X['PersonalityChanges'] + X['DifficultyCompletingTasks']
        )
        
        # Cholesterol ratios
        X['Chol_Ratio'] = X['CholesterolTotal'] / (X['CholesterolHDL'] + 1e-5)
        X['LDL_HDL_Ratio'] = X['CholesterolLDL'] / (X['CholesterolHDL'] + 1e-5)
        
        # Lifestyle score
        X['Lifestyle_Score'] = (
            X['PhysicalActivity'] + X['DietQuality'] + X['SleepQuality'] - 
            X['Smoking'] * 2 - X['AlcoholConsumption'] / 10
        )
        
        # Age groups
        X['Age_Group_Senior'] = (X['Age'] >= 75).astype(int)
        X['Age_Group_Elderly'] = (X['Age'] >= 85).astype(int)
        
        # BMI categories
        X['BMI_Overweight'] = ((X['BMI'] >= 25) & (X['BMI'] < 30)).astype(int)
        X['BMI_Obese'] = (X['BMI'] >= 30).astype(int)
        
        return X
    
    def _select_features(self, X, y, n_features=40):
        """Feature selection using Random Forest importance"""
        logging.info(f"Selecting top {n_features} features...")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X, y)
        
        # Get feature importances
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        selected_features = feature_importance.head(n_features)['feature'].tolist()
        
        logging.info(f"Top 10 most important features:")
        for i, row in feature_importance.head(10).iterrows():
            logging.info(f"{row['feature']}: {row['importance']:.4f}")
        
        return X[selected_features]
    
    def create_enhanced_models(self):
        """Create an ensemble of optimized models"""
        logging.info("Creating enhanced model ensemble...")
        
        # Base models with optimized hyperparameters
        models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.random_state
            )
        }
        
        # Enhanced Voting Classifier
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', models['rf']),
                ('xgb', models['xgb']),
                ('lgb', models['lgb']),
                ('svm', models['svm'])
            ],
            voting='soft'
        )
        
        # Enhanced Stacking Classifier
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', models['rf']),
                ('xgb', models['xgb']),
                ('lgb', models['lgb']),
                ('svm', models['svm']),
                ('mlp', models['mlp'])
            ],
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=5,
            stack_method='predict_proba'
        )
        
        self.models = {
            **models,
            'voting_enhanced': voting_clf,
            'stacking_enhanced': stacking_clf
        }
        
        return self.models
    
    def train_and_evaluate(self, X, y):
        """Train and evaluate all models"""
        logging.info("Training and evaluating models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train and evaluate each model
        results = {}
        
        for name, model in self.models.items():
            logging.info(f"Training {name}...")
            
            # Train model
            if name in ['svm', 'mlp']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob)
            }
            
            logging.info(f"{name} - Accuracy: {results[name]['accuracy']:.4f}, "
                        f"F1: {results[name]['f1']:.4f}, "
                        f"ROC-AUC: {results[name]['roc_auc']:.4f}")
        
        self.results = results
        return results
    
    def save_best_models(self):
        """Save the best performing models"""
        logging.info("Saving best models...")
        
        # Find best models
        best_voting = max([k for k in self.results.keys() if 'voting' in k], 
                         key=lambda x: self.results[x]['f1'])
        best_stacking = max([k for k in self.results.keys() if 'stacking' in k], 
                           key=lambda x: self.results[x]['f1'])
        
        # Save models
        joblib.dump(self.models[best_voting], 'alzheimers_voting_model_enhanced.pkl')
        joblib.dump(self.models[best_stacking], 'alzheimers_stacking_model_enhanced.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        
        logging.info(f"Saved best voting model: {best_voting}")
        logging.info(f"Saved best stacking model: {best_stacking}")
        
        return best_voting, best_stacking
    
    def generate_report(self):
        """Generate a comprehensive model performance report"""
        logging.info("Generating performance report...")
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        
        # Print report
        print("\n" + "="*80)
        print("ENHANCED ALZHEIMER'S DETECTION MODEL PERFORMANCE REPORT")
        print("="*80)
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nModel Performance Comparison:")
        print(results_df.to_string())
        
        # Best performers
        best_accuracy = results_df['accuracy'].idxmax()
        best_f1 = results_df['f1'].idxmax()
        best_roc_auc = results_df['roc_auc'].idxmax()
        
        print(f"\nBest Models:")
        print(f"Accuracy: {best_accuracy} ({results_df.loc[best_accuracy, 'accuracy']:.4f})")
        print(f"F1 Score: {best_f1} ({results_df.loc[best_f1, 'f1']:.4f})")
        print(f"ROC-AUC: {best_roc_auc} ({results_df.loc[best_roc_auc, 'roc_auc']:.4f})")
        
        # Recommendations
        print(f"\nRecommendations:")
        if 'stacking_enhanced' in results_df.index:
            stacking_f1 = results_df.loc['stacking_enhanced', 'f1']
            print(f"- Use Stacking Enhanced model for production (F1: {stacking_f1:.4f})")
        
        print("- Consider ensemble methods for better performance")
        print("- Monitor model performance on new data")
        print("- Retrain periodically with new data")
        
        return results_df

def main():
    """Main function to run enhanced model training"""
    try:
        # Initialize trainer
        trainer = EnhancedAlzheimerModelTrainer()
        
        # Load and preprocess data
        X, y = trainer.load_and_preprocess_data("dataset/alzheimers_disease_data.csv")
        
        # Create models
        trainer.create_enhanced_models()
        
        # Train and evaluate
        trainer.train_and_evaluate(X, y)
        
        # Save best models
        trainer.save_best_models()
        
        # Generate report
        trainer.generate_report()
        
        logging.info("Enhanced model training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise

if __name__ == "__main__":
    main()