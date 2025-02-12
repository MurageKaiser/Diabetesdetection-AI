from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from app.model import ModelLoader, preprocess_input

# Create a FastAPI router
router = APIRouter()

# Define request model (input structure)
class DiabetesRequest(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float = 0.5  # Default value
    age: int

# Load model
model = ModelLoader.get_diabetes_model(model_path="./app/model/diabetesdetection_model")

@router.post("/predict")
async def predict_diabetes(data: DiabetesRequest):
    """Endpoint to predict diabetes"""
    try:
        # Convert input into NumPy array for prediction
        processed_data = preprocess_input(data.dict())

        # Run model prediction
        prediction = model.predict(processed_data)
        result = int(prediction[0] > 0.5)  # Convert probability to binary 0 or 1

        return {"prediction": result, "confidence": float(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
