import sys, os, yaml
from typing import Tuple, Any
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    save_object
)
from src.entity.model_factory import ModelFactory, BestModelDetail
from src.entity.config_entity import ModelTrainerConfig, DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from src.target_mapping import InsuranceModel



import matplotlib
matplotlib.use('Agg')



class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(
        self, train: np.ndarray, test: np.ndarray
    ) -> Tuple[BestModelDetail, ClassificationMetricArtifact]:
        try:
            logging.info("Using ModelFactory to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)

            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train)
            y_test = encoder.transform(y_test)

            best_model_detail = model_factory.get_best_model(
                X_train=x_train,
                y_train=y_train,
                base_accuracy=self.model_trainer_config.expected_accuracy
            )

            # Predict on train and test to check overfitting
            y_train_pred = best_model_detail.best_model.predict(x_train)
            y_test_pred = best_model_detail.best_model.predict(x_test)

            f1_train = f1_score(y_train, y_train_pred)
            f1_test = f1_score(y_test, y_test_pred)

            # Check overfitting: if train score much higher than test score (threshold can be tuned)
            overfit_threshold = 0.6  # e.g. 10% difference allowed
            if (f1_train - f1_test) > overfit_threshold:
                raise Exception(f"Overfitting detected: F1 train={f1_train:.3f}, F1 test={f1_test:.3f}")

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1_test,
                precision_score=precision_score(y_test, y_test_pred),
                recall_score=recall_score(y_test, y_test_pred)
            )

            return best_model_detail, metric_artifact

        except Exception as e:
            logging.error(f"Error during model training and evaluation: {e}")
            raise e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path, 
                                            expected_type=Pipeline)


            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                raise Exception("No best model found with score more than base score")

            insurance_model = InsuranceModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model_detail.best_model
            )

            save_object(self.model_trainer_config.trained_model_file_path, insurance_model)
            
            # --- Save metrics YAML ---
            metrics_path = os.path.join(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                "metrics.yaml"
            )
            with open(metrics_path, "w") as f:
                yaml.dump(metric_artifact.__dict__, f)

            logging.info(f"Saved metrics artifact yaml at: {metrics_path}")

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                test_metric_artifact=metric_artifact
            )

        except Exception as e:
            print(e)


if __name__ == "__main__":
    
    data_transformation_config = DataTransformationConfig()
    data_transformation_artifacts = DataTransformationArtifact(
        transformed_object_file_path=data_transformation_config.transformed_object_file_path,
        transformed_train_file_path=data_transformation_config.transformed_train_file_path,
        transformed_test_file_path=data_transformation_config.transformed_test_file_path
    )
    
    model_trainer_config = ModelTrainerConfig()
    model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifacts,
                                 model_trainer_config=model_trainer_config)
    
    model_trainer_artifact = model_trainer.initiate_model_trainer()
    
    


