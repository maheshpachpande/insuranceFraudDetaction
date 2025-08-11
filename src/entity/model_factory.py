
git_remote_url = "https://github.com/maheshpachpande/insuranceFraudDetaction.git"
git_clone  = "https://github.com/maheshpachpande/insuranceFraudDetaction.git"

from src.logger import logging

import yaml
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Optional
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import dagshub
import os
import tempfile
from src.exception import CustomException

# ✅ Initialize DagsHub tracking (done once)
dagshub.init(repo_owner="pachpandemahesh300", repo_name="insuranceFraudDetaction.mlflow", mlflow=True)
mlflow.set_experiment("insurance-fraud-experiment")


@dataclass
class BestModelDetail:
    best_model: Any
    best_score: float
    best_parameters: dict


class ModelFactory:
    def __init__(self, model_config_path: str):
        self.model_config_path = model_config_path
        self.models_config = self._load_model_config()

    def _load_model_config(self) -> Dict:
        try:
            with open(self.model_config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model(self,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       base_accuracy: float = 0.6
                    ) -> BestModelDetail:

        problem_type = self._detect_problem_type(y_train)
        best_model: Optional[BaseEstimator] = None
        best_score = -np.inf
        best_params: dict = {}

        for model_name, model_info in self.models_config.items():
            with mlflow.start_run(run_name=model_name):
                try:
                    mlflow.log_params(model_info.get('params', {}))
                    mlflow.log_param("model_class", model_info['class'])

                    model_class = self._import_model_class(model_info['class'])
                    param_grid = model_info.get('params', {})
                    scoring = self._get_scoring(problem_type)

                    mlflow.sklearn.autolog(log_models=False)

                    search = GridSearchCV(
                        estimator=model_class(),
                        param_grid=param_grid,
                        scoring=scoring,
                        cv=5,
                        n_jobs=-1
                    )
                    search.fit(X_train, y_train)

                    mlflow.log_metric("cv_best_score", search.best_score_)

                    # Log the YAML config as artifact
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_config:
                        with open(self.model_config_path, 'r') as src:
                            temp_config.write(src.read().encode())
                        mlflow.log_artifact(temp_config.name, artifact_path="config")

                    if search.best_score_ > best_score and search.best_score_ >= base_accuracy:
                        best_score = search.best_score_
                        best_model = search.best_estimator_
                        best_params = search.best_params_

                    # ✅ Log best model to MLflow (DagsHub)
                    # mlflow.sklearn.log_model(search.best_estimator_, artifact_path="best_model")
                    try:
                        mlflow.sklearn.log_model(search.best_estimator_, name="best_model")
                    except Exception as e:
                        logging.warning(f"MLflow logging failed: {e}")


                    print(f"Model: {model_name}, Best Score: {search.best_score_}, Best Params: {search.best_params_}")

                except Exception as e:
                    raise CustomException(e, sys)

        if best_model is None:
            raise ValueError("No model met the base accuracy requirement.")

        return BestModelDetail(best_model, best_score, best_params)

    def _import_model_class(self, full_class_string: str):
        module_path, class_name = full_class_string.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    def _detect_problem_type(self, y: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(y):
            return 'regression'
        return 'classification'

    def _get_scoring(self, problem_type: str) -> str:
        return 'accuracy' if problem_type == 'classification' else 'r2'
