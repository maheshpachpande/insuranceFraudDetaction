import sys
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.utils.main_utils import read_yaml_file
from src.constants import SCHEMA_FILE_PATH
from pathlib import Path
from src.logger import logging


import yaml




STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        
        ingestion_cfg = DataIngestionConfig()

        ingestion = DataIngestion(ingestion_cfg)
        artifact = ingestion.initiate_data_ingestion()
        print(f"Data Ingestion completed: {artifact}")



if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        sys.exit(str(e))
