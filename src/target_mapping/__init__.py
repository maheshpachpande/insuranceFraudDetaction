import sys
from typing import Protocol
from sklearn.pipeline import Pipeline
from pandas import DataFrame
from src.logger import logging
from src.exception import CustomException



class TargetValueMapping:
    def __init__(self):
        self.neg: int = 0
        self.pos: int = 1

    def to_dict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))
    
    
    


class Predictable(Protocol):
    """Protocol for any model with a scikit-learn style predict method."""
    def predict(self, X) -> list:
        ...


class InsuranceModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: Predictable):
        """
        Production-grade USvisaModel wrapper for preprocessing and prediction.

        Args:
            preprocessing_object (Pipeline): Fitted preprocessing pipeline
            trained_model_object (Predictable): Fitted model object with a predict method
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame):
        """
        Transforms raw inputs using the preprocessing pipeline
        and performs predictions with the trained model.

        Args:
            dataframe (pd.DataFrame): Raw input dataframe

        Returns:
            list | np.ndarray: Predictions from the trained model
        """
        logging.info("Entered predict method of USvisaModel class")

        try:
            logging.info("Transforming input features")
            transformed_features = self.preprocessing_object.transform(dataframe)

            logging.info("Performing predictions")
            return self.trained_model_object.predict(transformed_features)

        except Exception as e:
            raise CustomException(e, sys) 

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return self.__repr__()