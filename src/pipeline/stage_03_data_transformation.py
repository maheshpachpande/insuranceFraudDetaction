import sys
import logging
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataTransformationConfig, DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.utils.main_utils import read_yaml_file


VALIDATION_STATUS = "artifact/data_validation/validation.yaml"



STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
        
    def main(self):
        try:
            status = read_yaml_file(VALIDATION_STATUS)
            
            data_validation_config = DataValidationConfig()
            data_validation_artifact = DataValidationArtifact(
                validation_status=status['validation_status'],
                valid_train_file_path=data_validation_config.valid_train_file_path,
                valid_test_file_path=data_validation_config.valid_test_file_path,
                drift_report_file_path=data_validation_config.drift_report_file_path
            )
            
            data_transformation_config = DataTransformationConfig()
            
            transformer = DataTransformation(data_validation_artifact, data_transformation_config)
            artifact = transformer.initiate_data_transformation()
            print(artifact)
            logging.info(f"Data transformation completed Artifact: {artifact}")
        except Exception as e:
            logging.error(f"Error in data transformation stage: {e}")
            sys.exit(1)



if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        sys.exit(str(e))
