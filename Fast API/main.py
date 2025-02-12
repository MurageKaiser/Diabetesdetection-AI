from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.model import ModelLoader, preprocess_input

app = FastAPI(title="Diabetes Prediction API", description="Predicts diabetes based on input features", version="1.0")

model = ModelLoader.get_diabetes_model(model_path='./app/model/detectionmodel.keras_FILES')

class DiabetesInput(BaseModel):
    gender: str
    age: int
    bmi: float
    glucose: float
    insulin: float
    blood_pressure: float
    pregnancies: Optional[int] = None
    skin_thickness_category: str  # Must be a valid category

    # Family history questions (Mandatory, no default values)
    mother_diabetes: bool
    father_diabetes: bool
    sibling_diabetes: bool
    grandparent_diabetes: bool
    early_diagnosis: bool  # Family member diagnosed before 40

# Define valid categories for skin thickness
SKIN_THICKNESS_MAP = {
    "Thin": 12,  
    "Average": 20,
    "Overweight": 30
}

def calculate_pedigree(mother, father, sibling, grandparent, early_diag):
    """
    Estimate the Diabetes Pedigree Function (DPF) based on family history.
    """
    pedigree_score = 0.2  # Base score

    if mother: pedigree_score += 0.3
    if father: pedigree_score += 0.3
    if sibling: pedigree_score += 0.2
    if grandparent: pedigree_score += 0.1
    if early_diag: pedigree_score += 0.4

    return round(pedigree_score, 2)  # Round to 2 decimal places

@app.post("/predict/")
def predict(data: DiabetesInput):
    """
    Endpoint to predict diabetes based on user input.
    """
    try:
        input_dict = data.dict()

        # Handle gender logic
        if input_dict["gender"].lower() == "male":
            if input_dict["pregnancies"] != 0:
                input_dict["pregnancies"] = 0
                app.logger.info("Pregnancies value overridden to 0 for male user.")

        elif input_dict["pregnancies"] is None:
            raise HTTPException(status_code=400, detail="Pregnancies must be provided for female users.")

        # Validate skin thickness category
        if input_dict["skin_thickness_category"] not in SKIN_THICKNESS_MAP:
            raise HTTPException(status_code=400, detail="Invalid skin thickness category. Choose from: Thin, Average, Overweight.")

        # Convert skin thickness category to numerical value
        input_dict["skin_thickness"] = SKIN_THICKNESS_MAP[input_dict.pop("skin_thickness_category")]

        # Calculate DPF from user-provided answers (no default)
        input_dict["diabetes_pedigree_function"] = calculate_pedigree(
            input_dict["mother_diabetes"], 
            input_dict["father_diabetes"], 
            input_dict["sibling_diabetes"], 
            input_dict["grandparent_diabetes"], 
            input_dict["early_diagnosis"]
        )

        # Preprocess input
        processed_input = preprocess_input(input_dict)

        # Make prediction
        prediction = model.predict(processed_input)
        result = "Diabetic" if prediction[0] > 0.5 else "Non-Diabetic"

        return {
            "prediction": result,
            "confidence": float(prediction[0]),
            "calculated_dpf": input_dict["diabetes_pedigree_function"],
            "note": "Pregnancies value was automatically set to 0 for male users."
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/health/")
def health_check():
    return {"status": "API is healthy!"}
