from src.entity.config_entity import ModelEvaluationConfig, DataIngestionConfig, ModelTrainerConfig
from src.entity.artifact_entity import ClassificationMetricArtifact, ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import CustomException
from src.constants import TARGET_COLUMN
from src.logger import logging
import sys, json, os
from dataclasses import asdict
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import S3_InsuranceEstimator
from dataclasses import dataclass
from src.target_mapping import TargetValueMapping, InsuranceModel
from src.exception import CustomException
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.utils.main_utils import read_yaml_file
from dotenv import load_dotenv
load_dotenv()

from typing import Optional

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float



class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) 

    def get_best_model(self) -> Optional[S3_InsuranceEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            usvisa_estimator = S3_InsuranceEstimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if usvisa_estimator.is_model_present(model_path=model_path):
                return usvisa_estimator
            return None
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            raise CustomException(e, exc_tb)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            y = y.replace(
                TargetValueMapping().to_dict()
            )

            # trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_f1_score = self.model_trainer_artifact.test_metric_artifact.f1_score

            best_model_f1_score=0.0
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)

                
                if y.dtype == object or isinstance(y[0], str):
                    encoder = LabelEncoder()
                    y = encoder.fit_transform(y)  # N → 0, Y → 1

                # Ensure y_pred is also integer
                y_hat_best_model = np.array(y_hat_best_model, dtype=int)
                
                best_model_f1_score = f1_score(y, y_hat_best_model)

                best_model_f1_score = float(best_model_f1_score)
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=float(trained_model_f1_score) > float(tmp_best_model_score),
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            raise CustomException(e, exc_tb)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation

        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            # Ensure the directory exists (not the file itself)
            artifact_file_path = self.model_eval_config.artifact_file_path
            os.makedirs(os.path.dirname(artifact_file_path), exist_ok=True)

            # Save artifact to JSON
            with open(artifact_file_path, 'w') as f:
                json.dump(asdict(model_evaluation_artifact), f, indent=4)

            logging.info(f"Model evaluation artifact saved at: {artifact_file_path}")
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            raise CustomException(e, exc_tb)


if __name__ == '__main__':
    
    metric_path = "artifact/model_trainer/trained_model/metrics.yaml"
    metrics = read_yaml_file(metric_path)
    
    from dotenv import load_dotenv
    load_dotenv()

    
    model_eval_config = ModelEvaluationConfig()
    
    data_ingestion_config = DataIngestionConfig()
    data_ingestion_artifact = DataIngestionArtifact(
        raw_file_path=data_ingestion_config.feature_store_file_path,
        trained_file_path=data_ingestion_config.training_file_path,
        test_file_path=data_ingestion_config.testing_file_path
    )

    model_trainer_config = ModelTrainerConfig()
            
    test_metric_artifact = ClassificationMetricArtifact(
                            f1_score=metrics.get("f1_score", 0.0),
                            precision_score=metrics.get("precision_score", 0.0),
                            recall_score=metrics.get("recall_score", 0.0),
                        )

            
    model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=model_trainer_config.trained_model_file_path,
            test_metric_artifact=test_metric_artifact
        )
        
    model_evaluation_config = ModelEvaluationConfig()
    model_evaluation_artifact = ModelEvaluation(
        model_eval_config=model_evaluation_config,
        data_ingestion_artifact=data_ingestion_artifact,
        model_trainer_artifact=model_trainer_artifact
    )
    model_evaluation_artifact.evaluate_model()
