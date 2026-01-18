import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pickle  # Changed from dill to pickle
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save object to file using pickle
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        print(f"✅ Object saved to {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load object from file
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models and return R2 scores
    """
    try:
        report = {}

        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            
            print(f"✅ {model_name}: Train R2={train_model_score:.4f}, Test R2={test_model_score:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)  # Don't use 'pass' - raise the error!