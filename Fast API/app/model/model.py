import tensorflow as tf
import numpy as np
import io
import logging
from fastapi import HTTPException
import os
from tensorflow.keras.models import load_model

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configure basic logging
def configure_logging():
    """Configure logging for the diabetes detection app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("diabetes_model.log", mode="a")
        ]
    )

# Preprocess input data function
def preprocess_input(data: dict) -> np.ndarray:
    """Preprocess input data for diabetes prediction."""
    try:
        required_features = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                             'insulin', 'bmi', 'age']
        optional_features = ['diabetes_pedigree_function']
        
        # Check if all required features are provided
        for feature in required_features:
            if feature not in data:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Set default for optional feature
        if 'diabetes_pedigree_function' not in data:
            logger.warning("Diabetes Pedigree Function not provided. Using default value of 0.5.")
            data['diabetes_pedigree_function'] = 0.5  

        all_features = required_features + optional_features
        input_array = np.array([data[feature] for feature in all_features], dtype=np.float32)
        
        normalization_factors = [10, 200, 120, 100, 300, 50, 100, 2] 
        input_array = input_array / np.array(normalization_factors, dtype=np.float32)
        
        return np.expand_dims(input_array, axis=0)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

# Model Loader Class
class ModelLoader:
    _diabetes_model = None

    @classmethod
    def get_diabetes_model(cls, model_path: str = './app/model/detectionmodel.keras_FILES'):
        """Load the Diabetes Keras model."""
        if cls._diabetes_model is None:
            try:
                logger.info(f"Loading Diabetes model from {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                cls._diabetes_model = load_model(model_path)  
                logger.info("Diabetes model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load Diabetes model: {e}")
                raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")
        
        return cls._diabetes_model

# Prediction Logic
def predict_with_model(model, input_data: dict, class_names: list) -> dict:
    """Make a prediction using the diabetes detection model."""
    try:
        processed_input = preprocess_input(input_data)
        
        signature = model.signatures['serving_default']
        predictions = signature(tf.convert_to_tensor(processed_input))
        output_key = list(signature.structured_outputs.keys())[0]
        predictions = predictions[output_key].numpy()
        
        predicted_class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_index])
        predicted_class = class_names[predicted_class_index]

        if confidence < 0.30:
            description = "Low likelihood of diabetes."
        elif confidence < 0.75:
            description = "Moderate risk. Consider additional tests."
        else:
            description = "High risk. Immediate medical evaluation recommended."

        return {
            "confidence": confidence,
            "predicted_class": predicted_class,
            "description": description
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Expose only available model
get_diabetes_model = ModelLoader.get_diabetes_model
