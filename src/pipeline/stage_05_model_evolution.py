import sys
import logging
from src.components.model_evaluation import ModelEvaluation
from src.entity.config_entity import ModelTrainerConfig, DataTransformationConfig, DataIngestionConfig, ModelEvaluationConfig
from src.entity.artifact_entity import ClassificationMetricArtifact,DataTransformationArtifact, ModelTrainerArtifact, DataIngestionArtifact
from src.utils.main_utils import read_yaml_file

metric_path = "artifact/model_trainer/trained_model/metrics.yaml"
metrics = read_yaml_file(metric_path)

STAGE_NAME = "Model Evolution stage"

class ModelEvolutionTrainingPipeline:
    def __init__(self):
        pass
    
    
    def main(self):
        try:
            data_ingestion_config = DataIngestionConfig()
            data_ingestion_artifact = DataIngestionArtifact(
                raw_file_path=data_ingestion_config.feature_store_file_path,
                trained_file_path=data_ingestion_config.training_file_path,
                test_file_path=data_ingestion_config.testing_file_path
            )
            
            
            data_transformation_config = DataTransformationConfig()
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=data_transformation_config.transformed_test_file_path
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
        except Exception as e:
            logging.error(f"Error during model training and evaluation: {e}")
            raise e
            
if __name__ == "__main__":
    try:
        model_evolution_pipeline = ModelEvolutionTrainingPipeline()
        model_evolution_pipeline.main()
    except Exception as e:
        logging.error(f"Error during model training and evaluation: {e}")
        sys.exit(1)
            
            