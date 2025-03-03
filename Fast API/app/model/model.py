import tensorflow as tf
import numpy as np
import logging
from fastapi import HTTPException
import os
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("diabetes_model.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

def preprocess_input(data: dict) -> np.ndarray:
    """Preprocess input data for diabetes prediction."""
    try:
        logger.info("Received input data: %s", data)
        
        required_inputs = ['age', 'glucose', 'blood_pressure', 'bmi', 'insulin', 
                           'family_history', 'gender', 'skin_thickness', 'pregnancies']
        
        for feature in required_inputs:
            if feature not in data:
                raise ValueError(f"Missing required input: {feature}")
        
        gender = data["gender"].strip().lower()
        if gender not in ["male", "female"]:
            raise ValueError(f"Invalid gender: {data['gender']} (Expected: 'male' or 'female')")
        
        pregnancies = data["pregnancies"] if gender == "female" else 0
        skin_thickness = data["skin_thickness"]
        diabetes_pedigree_function = 0.8 if data["family_history"] else 0.2  
        
        input_values = [
            pregnancies,
            data["glucose"],
            data["blood_pressure"],
            skin_thickness,
            data["insulin"],
            data["bmi"],
            data["age"],
            diabetes_pedigree_function
        ]
        
        normalization_factors = [10, 200, 120, 100, 300, 50, 100, 2] 
        input_array = np.array(input_values, dtype=np.float32)
        input_array /= np.array(normalization_factors, dtype=np.float32)
        
        return np.expand_dims(input_array, axis=0)
    
    except Exception as e:
        logger.error("Preprocessing error: %s", str(e))
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

class ModelLoader:
    _diabetes_model = None

    @classmethod
    def get_diabetes_model(cls, model_path: str = './app/model/Diadetection.keras_FILES'):
        """Load the Diabetes Keras model."""
        if cls._diabetes_model is None:
            try:
                logger.info("Loading Diabetes model from %s", model_path)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                cls._diabetes_model = load_model(model_path)  
                logger.info("Diabetes model loaded successfully")
            
            except Exception as e:
                logger.error("Failed to load Diabetes model: %s", str(e))
                raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")
        
        return cls._diabetes_model

# Expose only available model
get_diabetes_model = ModelLoader.get_diabetes_model
