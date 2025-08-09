import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Optional
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator


@dataclass
class BestModelDetail:
    best_model: Any
    best_score: float
    best_parameters: dict


class ModelFactory:
    def __init__(self, model_config_path: str):
        """
        Custom Model Factory to train and select the best model based on YAML configuration.
        """
        self.model_config_path = model_config_path
        self.models_config = self._load_model_config()

    def _load_model_config(self) -> Dict:
        """Load YAML model config file."""
        try:
            with open(self.model_config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model config file not found at: {self.model_config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def get_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        base_accuracy: float = 0.6
    ) -> BestModelDetail:
        """
        Train multiple models from config and return the one with the best CV score.
        """
        problem_type = self._detect_problem_type(y_train)
        best_model: Optional[BaseEstimator] = None
        best_score = -np.inf
        best_params: dict = {}

        for model_name, model_info in self.models_config.items():
            try:
                model_class = self._import_model_class(model_info['class'])
                print(f"Training model: {model_name} with class {model_info['class']}")
                param_grid = model_info.get('params', {})

                scoring = self._get_scoring(problem_type)
                search = GridSearchCV(
                    estimator=model_class(),
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=5,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)

                if search.best_score_ > best_score and search.best_score_ >= base_accuracy:
                    best_score = search.best_score_
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                    
                print(f"Model: {model_name}, Best Score: {search.best_score_}, Best Params: {search.best_params_}")

            except Exception as e:
                print(f"Error training model {model_name}: {e}")

        if best_model is None:
            raise ValueError("No model met the base accuracy requirement.")

        return BestModelDetail(best_model, best_score, best_params)

    def _import_model_class(self, full_class_string: str):
        """Dynamically import a model class from string path."""
        module_path, class_name = full_class_string.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    def _detect_problem_type(self, y: pd.Series) -> str:
        """Detect if problem is classification or regression."""
        if pd.api.types.is_numeric_dtype(y):
            return 'regression'
        return 'classification'

    def _get_scoring(self, problem_type: str) -> str:
        """Return scoring metric based on problem type."""
        return 'accuracy' if problem_type == 'classification' else 'r2'



# if __name__ == "__main__":
#     # Example: path to YAML model config
#     model_config_path = "config/model.yaml"

#     # Create dummy classification data
#     from sklearn.datasets import make_classification
#     X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)

#     # Convert to DataFrame/Series (optional but matches your type hints)
#     import pandas as pd
#     X_df = pd.DataFrame(X)
#     y_series = pd.Series(y)

#     # Initialize and run ModelFactory
#     factory = ModelFactory(model_config_path=model_config_path)
#     try:
#         best_model_detail = factory.get_best_model(
#             X_train=X_df,
#             y_train=y_series,
#             base_accuracy=0.6
#         )
#         print("Best model:", best_model_detail.best_model)
#         print("Best score:", best_model_detail.best_score)
#         print("Best parameters:", best_model_detail.best_parameters)
#     except Exception as e:
#         print("Error:", e)
