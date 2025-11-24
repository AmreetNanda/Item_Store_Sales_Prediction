import os
import sys
import pandas as pd
from src.Exception import CustomException
from src.Utils import load_object

class CustomData:
    """
    A class to structure user input data into a DataFrame for prediction.
    """
    def __init__(
        self,
        Item_Weight: float,
        Item_Visibility: float,
        Item_MRP: float,
        Outlet_Establishment_Year: int,
        Item_Fat_Content: str,
        Item_Type: str,
        Outlet_Size: str,
        Outlet_Location_Type: str,
        Outlet_Type: str
        # Item_Identifier: str,
        # Outlet_Identifier: str
    ):
        self.Item_Weight = Item_Weight
        self.Item_Visibility = Item_Visibility
        self.Item_MRP = Item_MRP
        self.Outlet_Establishment_Year = Outlet_Establishment_Year
        self.Item_Fat_Content = Item_Fat_Content
        self.Item_Type = Item_Type
        self.Outlet_Size = Outlet_Size
        self.Outlet_Location_Type = Outlet_Location_Type
        self.Outlet_Type = Outlet_Type
        # self.Item_Identifier = Item_Identifier
        # self.Outlet_Identifier = Outlet_Identifier

    def get_data_as_dataframe(self):
        """
        Convert input attributes to a pandas DataFrame and add derived columns.
        """
        try:
            data_dict = {
                "Item_Weight": [self.Item_Weight],
                "Item_Visibility": [self.Item_Visibility],
                "Item_MRP": [self.Item_MRP],
                "Outlet_Establishment_Year": [self.Outlet_Establishment_Year],
                "Item_Fat_Content": [self.Item_Fat_Content],
                "Item_Type": [self.Item_Type],
                "Outlet_Size": [self.Outlet_Size],
                "Outlet_Location_Type": [self.Outlet_Location_Type],
                "Outlet_Type": [self.Outlet_Type]
                # "Item_Identifier": [self.Item_Identifier],
                # "Outlet_Identifier": [self.Outlet_Identifier]
            }

            df = pd.DataFrame(data_dict)

            # Derived columns to match training preprocessing
            # df['Item_Identifier_Categories'] = df['Item_Identifier'].str[:2]
            df['Outlet_Age'] = 2023 - df['Outlet_Establishment_Year']

            return df

        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    """
    A class to load the preprocessing object and trained model,
    and make predictions.
    """
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.model_path = os.path.join("artifacts", "model.pkl")

    def predict(self, features: pd.DataFrame):
        """
        Predict the sales for the input features.
        """
        try:
            # Load preprocessor and trained model
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            # Ensure all required columns exist
            # required_cols = [
            #     'Item_Weight','Item_Visibility','Item_MRP','Item_Fat_Content',
            #     'Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type',
            #     'Item_Identifier_Categories','Outlet_Age','Outlet_Identifier'
            # ]
            required_cols = [
                'Item_Weight','Item_Visibility','Item_MRP','Item_Fat_Content',
                'Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type','Outlet_Age'
            ]
            for col in required_cols:
                if col not in features.columns:
                    raise CustomException(f"Missing required column: {col}", sys)

            # Transform features using preprocessor
            features_transformed = preprocessor.transform(features)

            # Make predictions
            predictions = model.predict(features_transformed)
            return predictions

        except Exception as e:
            raise CustomException(e, sys)
