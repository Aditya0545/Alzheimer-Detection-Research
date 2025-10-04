#!/usr/bin/env python3
"""
Setup script for Alzheimer's Detection App
Installs required Python dependencies and checks model files
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description} found: {filepath}")
        return True
    else:
        print(f"❌ {description} missing: {filepath}")
        return False

def main():
    print("🧠 Alzheimer's Detection App Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    else:
        print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install Python dependencies
    python_deps = [
        "scikit-learn",
        "numpy", 
        "joblib",
        "pandas"
    ]
    
    for dep in python_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  Failed to install {dep}. You may need to install it manually.")
    
    # Check for model files
    print("\n🔍 Checking for model files...")
    voting_model = "alzheimers_voting_model.pkl"
    stacking_model = "alzheimers_stacking_model.pkl"
    
    models_found = True
    if not check_file_exists(voting_model, "Voting model"):
        models_found = False
    if not check_file_exists(stacking_model, "Stacking model"):
        models_found = False
    
    if not models_found:
        print("\n⚠️  Model files not found!")
        print("Please ensure you have the following files in the project root:")
        print("- alzheimers_voting_model.pkl")
        print("- alzheimers_stacking_model.pkl")
        print("\nYou can copy them from your original project directory.")
    
    # Test Python script
    print("\n🧪 Testing Python ML script...")
    if os.path.exists("ml_predictor.py"):
        print("✅ ml_predictor.py found")
    else:
        print("❌ ml_predictor.py not found")
    
    print("\n📋 Next Steps:")
    print("1. Install Node.js dependencies: npm install")
    print("2. Start the development server: npm run dev")
    print("3. Open http://localhost:3000 in your browser")
    
    print("\n🎉 Setup completed!")

if __name__ == "__main__":
    main()
