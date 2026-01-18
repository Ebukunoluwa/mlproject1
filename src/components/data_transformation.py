from dataclasses import dataclass
import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function creates the preprocessing pipeline
        for numerical and categorical features
        """
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False))
                ]
            )

            print(f"✅ Categorical columns: {categorical_columns}")
            print(f"✅ Numerical columns: {numerical_columns}")

            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function performs data transformation on train and test data
        """
        try:
            print("=== Reading train and test data ===")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print(f"Train data shape: {train_df.shape}")
            print(f"Test data shape: {test_df.shape}")

            print("=== Obtaining preprocessing object ===")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            print("=== Applying preprocessing object ===")

            # Transform the data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target
            train_arr = np.c_[
                input_feature_train_arr, 
                np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, 
                np.array(target_feature_test_df)
            ]

            print(f"✅ Train array shape: {train_arr.shape}")
            print(f"✅ Test array shape: {test_arr.shape}")

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test the transformation
    obj = DataTransformation()
    train_arr, test_arr, preprocessor_path = obj.initiate_data_transformation(
        "artifacts/train.csv",
        "artifacts/test.csv"
    )
    print(f"\n✅ Transformation complete!")
    print(f"Train array shape: {train_arr.shape}")
    print(f"Test array shape: {test_arr.shape}")
    print(f"Preprocessor saved at: {preprocessor_path}")