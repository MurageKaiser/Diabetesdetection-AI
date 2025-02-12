import os
import numpy as np
from model import ModelLoader, preprocess_input
from fastapi import HTTPException

print("Test script is running...")  # Confirm the script starts

# Test data for prediction (you can replace this with real input)
test_data = {
    "pregnancies": 2,
    "glucose": 120,
    "blood_pressure": 70,
    "skin_thickness": 20,
    "insulin": 80,
    "bmi": 30.5,
    "diabetes_pedigree_function": 0.5,
    "age": 25
}

# Step 1: Test model loading
try:
    print("Loading model...")  # Debug print
    model = ModelLoader.get_diabetes_model(model_path='./app/model/diabetesdetection_model')
    print("Model loaded successfully!")  # Debug print
except HTTPException as e:
    print(f"Error loading model: {e.detail}")
    exit(1)

# Step 2: Test preprocessing
try:
    print("Preprocessing input...")  # Debug print
    preprocessed_input = preprocess_input(test_data)
    print("Input data preprocessed successfully!")  # Debug print
except HTTPException as e:
    print(f"Error preprocessing input: {e.detail}")
    exit(1)

# Step 3: Test prediction (if the model and input are loaded correctly)
try:
    print("Making prediction...")  # Debug print
    predictions = model.predict(preprocessed_input)
    print(f"Prediction: {predictions}")
except Exception as e:
    print(f"Prediction failed: {e}")
