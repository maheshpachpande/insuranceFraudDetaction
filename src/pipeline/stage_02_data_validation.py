import sys
from src.components.data_validation import DataValidation
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig, DataIngestionConfig
from src.logger import logging


STAGE_NAME = "Data Validatation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
        
    def main(self):
        try:
            # Setup Data Ingestion Artifact based on previous step outputs
            data_ingestion_config = DataIngestionConfig()
            data_ingestion_artifact = DataIngestionArtifact(
                raw_file_path=data_ingestion_config.feature_store_file_path,
                trained_file_path=data_ingestion_config.training_file_path,
                test_file_path=data_ingestion_config.testing_file_path
            )

            # Prepare Data Validation Config
            data_validation_cfg = DataValidationConfig()

            validator = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_cfg
            )

            artifact = validator.initiate_data_validation()
            logging.info(f"Data validation completed. Artifact: {artifact}")

        except Exception as e:
            logging.error(f"Error in data validation stage: {e}")
            sys.exit(1)

if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        sys.exit(str(e))
