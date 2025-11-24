import sys
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.Exception import CustomException
from src.Logger import logging
from src.Utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # -----------------------------------------------------------
    # Function to detect outliers (IQR Method)
    # -----------------------------------------------------------
    def detect_outliers(self, df, feature):
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR

        return upper, lower

    # -----------------------------------------------------------
    # Create preprocessing pipelines
    # -----------------------------------------------------------
    def get_data_transformer_object(self):

        try:
            numerical_columns = [
                "Item_Weight",
                "Item_Visibility",
                "Item_MRP",
                "Outlet_Age"
            ]

            categorical_columns = [
                # "Item_Identifier_Categories",
                "Item_Fat_Content",
                "Item_Type",
                # "Outlet_Identifier",
                "Outlet_Size",
                "Outlet_Location_Type",
                "Outlet_Type"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessor object created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------------------------------------
    # Main Function: Apply Transformations
    # -----------------------------------------------------------
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # ------------------------------
            # LOAD DATA
            # ------------------------------
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Training and testing data loaded")

            # ======================================================
            # 1. CLEANING & FEATURE ENGINEERING FOR TRAINING DATA
            # ======================================================

            # Fill missing values
            train_df["Outlet_Size"] = train_df["Outlet_Size"].fillna(
                train_df["Outlet_Size"].mode()[0]
            )
            train_df["Item_Weight"] = train_df["Item_Weight"].fillna(
                train_df["Item_Weight"].mean()
            )

           # Outlier handling for Item_Visibility → use clipping, not row removal
            upper, lower = self.detect_outliers(train_df, "Item_Visibility")
            train_df["Item_Visibility"] = train_df["Item_Visibility"].clip(lower, upper)

            # Outlier handling for Item_Outlet_Sales (target)
            upper, lower = self.detect_outliers(train_df, "Item_Outlet_Sales")
            train_df["Item_Outlet_Sales"] = train_df["Item_Outlet_Sales"].clip(lower, upper)

            # Fix Item_Fat_Content
            train_df["Item_Fat_Content"] = train_df["Item_Fat_Content"].map(
                {
                    "Low Fat": "Low Fat",
                    "low fat": "Low Fat",
                    "LF": "Low Fat",
                    "Regular": "Regular",
                    "reg": "Regular",
                }
            )

            # Feature Engineering
            train_df["Outlet_Age"] = 2023 - train_df["Outlet_Establishment_Year"]
            del train_df["Outlet_Establishment_Year"]

            train_df["Outlet_Size"] = (
                train_df["Outlet_Size"]
                .map({"Small": 1, "Medium": 2, "High": 3})
                .fillna(1)
                .astype(int)
            )

            train_df["Outlet_Location_Type"] = train_df[
                "Outlet_Location_Type"
            ].str[-1:].astype(int)

            train_df["Item_Identifier_Categories"] = train_df[
                "Item_Identifier"
            ].str[0:2]

            logging.info("Training dataset cleaned successfully")

            # ======================================================
            # 2. APPLY SAME CLEANING TO TEST DATA
            # ======================================================

            test_df["Outlet_Size"] = test_df["Outlet_Size"].fillna(
                train_df["Outlet_Size"].mode()[0]
            )
            test_df["Item_Weight"] = test_df["Item_Weight"].fillna(
                train_df["Item_Weight"].mean()
            )

            test_df["Item_Fat_Content"] = test_df["Item_Fat_Content"].map(
                {
                    "Low Fat": "Low Fat",
                    "low fat": "Low Fat",
                    "LF": "Low Fat",
                    "Regular": "Regular",
                    "reg": "Regular",
                }
            )

            test_df["Outlet_Age"] = 2023 - test_df["Outlet_Establishment_Year"]
            del test_df["Outlet_Establishment_Year"]

            test_df["Outlet_Size"] = (
                test_df["Outlet_Size"]
                .map({"Small": 1, "Medium": 2, "High": 3})
                .fillna(train_df["Outlet_Size"].mode()[0])
                .astype(int)
            )

            test_df["Outlet_Location_Type"] = test_df[
                "Outlet_Location_Type"
            ].str[-1:].astype(int)

            test_df["Item_Identifier_Categories"] = test_df[
                "Item_Identifier"
            ].str[0:2]

            logging.info("Test dataset cleaned successfully")

            # ======================================================
            # 3. DEFINE TARGET AND DROP COLUMNS
            # ======================================================
            target_column_name = "Item_Outlet_Sales"
            drop_columns = [target_column_name, 'Item_Identifier', 'Outlet_Identifier']


            X_train = train_df.drop(columns=drop_columns)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=drop_columns)
            y_test = test_df[target_column_name]

            logging.info("Separated features and target variable")

            # ======================================================
            # 4. APPLY PREPROCESSING
            # ======================================================
            preprocessing_obj = self.get_data_transformer_object()

            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_transformed, y_train]
            test_arr = np.c_[X_test_transformed, y_test]

            # Save preprocessor
            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessing_obj,
            )

            logging.info("Preprocessing complete and object saved")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
