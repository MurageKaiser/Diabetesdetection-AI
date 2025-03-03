from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np
import logging
from app.model import ModelLoader, preprocess_input

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Diabetes Prediction API",
    description="Predicts diabetes based on input features",
    version="1.0"
)

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load trained model
model = ModelLoader.get_diabetes_model(model_path='./app/model/Diadetection.keras_FILES')

# Define valid categories for skin thickness
SKIN_THICKNESS_MAP = {
    "thin": 12,
    "average": 20,
    "thick": 30
}

class DiabetesInput(BaseModel):
    gender: Literal["male", "female"]
    age: int = Field(..., ge=0, description="Age must be a non-negative integer.")
    bmi: float = Field(..., ge=0, description="BMI must be non-negative.")
    glucose: float = Field(..., ge=0, description="Glucose level must be non-negative.")
    insulin: float = Field(..., ge=0, description="Insulin level must be non-negative.")
    blood_pressure: float = Field(..., ge=0, description="Blood pressure must be non-negative.")
    pregnancies: Optional[int] = Field(None, ge=0, description="Number of pregnancies (only for females).")
    skin_thickness_category: str
    family_history: bool  

def map_pedigree(family_history: bool) -> float:
    """Assigns a Diabetes Pedigree Function (DPF) score based on family history."""
    return 0.6 if family_history else 0.2  

@app.post("/predict/")
def predict(data: DiabetesInput):
    """Predict diabetes based on user input."""
    try:
        input_dict = data.dict()

        # Ensure pregnancies is set to 0 for males
        if input_dict["gender"] == "male":
            input_dict["pregnancies"] = 0

        # Normalize and validate skin thickness category
        skin_category = input_dict["skin_thickness_category"].strip().lower()
        if skin_category not in SKIN_THICKNESS_MAP:
            valid_options = ', '.join([k.capitalize() for k in SKIN_THICKNESS_MAP.keys()])
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid skin_thickness_category: '{skin_category.capitalize()}'. Expected one of: {valid_options}."
            )

        input_dict["skin_thickness"] = SKIN_THICKNESS_MAP[skin_category]
        del input_dict["skin_thickness_category"]  # Remove the category field

        # Compute Diabetes Pedigree Function
        input_dict["diabetes_pedigree_function"] = map_pedigree(input_dict["family_history"])

        logger.info("Processed input before prediction: %s", input_dict)

        # Preprocess input and make prediction
        processed_input = preprocess_input(input_dict)
        prediction = model.predict(processed_input)
        confidence = float(np.array(prediction).flatten()[0])  # Ensure compatibility with NumPy

        result = "Diabetic" if confidence > 0.5 else "Non-Diabetic"

        # Construct response object
        response = {
            "prediction": result,
            "confidence": confidence,
            "calculated_dpf": input_dict["diabetes_pedigree_function"],
        }

        # ✅ Only include the note for male users
        if input_dict["gender"] == "male":
            response["note"] = "Pregnancies was set to 0 for male users."

        return response

    except HTTPException as e:
        logger.error("Validation error: %s", e.detail)
        raise e
    except Exception as e:
        logger.error("Internal Server Error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/health/")
def health_check():
    return {"status": "API is healthy!"}
