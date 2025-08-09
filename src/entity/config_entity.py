import os
from src.constants import *
from dataclasses import dataclass


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = ARTIFACT_DIR


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, RAW_DATA_FILE)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_DATA_FILE)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_DATA_FILE)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = COLLECTION_NAME
    
    
    
@dataclass
class DataValidationConfig:        
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR)
    valid_data_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_VALID_DIR)
    valid_train_file_path: str = os.path.join(valid_data_dir, TRAIN_DATA_FILE)
    valid_test_file_path: str = os.path.join(valid_data_dir, TEST_DATA_FILE)
    prior_drift_report_file_path: str = os.path.join(data_validation_dir,
                                            DATA_VALIDATION_DRIFT_REPORT_DIR,
                                            DATA_VALIDATION_PRIOR_REPORT_FILE)
    drift_report_file_path: str = os.path.join(data_validation_dir,
                                            DATA_VALIDATION_DRIFT_REPORT_DIR,
                                                DATA_VALIDATION_REPORT_FILE)



@dataclass
class DataTransformationConfig:
        data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR )
        transformed_train_file_path: str = os.path.join( data_transformation_dir,DATA_TRANSFORMATION_OUTPUT_DIR,
        TRAIN_DATA_FILE.replace("csv", "npy"),)
        transformed_test_file_path: str = os.path.join(data_transformation_dir,  DATA_TRANSFORMATION_OUTPUT_DIR,
        TEST_DATA_FILE.replace("csv", "npy"), )
        transformed_object_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_OBJECT_DIR,
        PREPROCESSOR_FILE_NAME,)

