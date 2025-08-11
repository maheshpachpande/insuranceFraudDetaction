import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import CustomException
from src.logger import logging
from src.data_access.mongoDBToPandas import InsuranceData
from typing import Optional
from src.utils.main_utils import read_yaml_file
from src.constants import SCHEMA_FILE_PATH




class DataIngestion:
    def __init__(self, data_ingestion_config: Optional[DataIngestionConfig] = None) -> None:
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            if data_ingestion_config is None:
                data_ingestion_config = DataIngestionConfig()
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)


    def export_data_into_feature_store(self)->DataFrame:
        """
        Output      :   data is returned as artifact of data ingestion components
        """
        try:
            logging.info("Exporting data from mongodb")
            my_data = InsuranceData()
            dataframe = my_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            
            
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            
            if dataframe.empty:
                raise CustomException("Dataframe is empty. Please check the MongoDB collection.", sys)
            
            logging.info("creating directory for feature store file path")
            feature_store_file_path  = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            
            return dataframe

        except Exception as e:
            raise CustomException(e,sys)
        
        
    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Split the dataframe into train and test sets after dropping unnecessary columns.

        Args:
            dataframe (pd.DataFrame): The full dataset to split.
        """
        logging.info("Entered split_data_as_train_test method of DataIngestion class")

        try:
            # Drop unwanted columns based on schema config
            drop_columns = self._schema_config.get("drop_columns", [])
            if drop_columns:
                logging.info(f"Dropping columns: {drop_columns}")
                dataframe = dataframe.drop(columns=drop_columns, errors="ignore")

            # Perform train-test split
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42  # for reproducibility
            )

            # Create directories for train and test file paths
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Save train and test datasets
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info(f"Exported train and test datasets to {dir_path}")
            logging.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")

        except Exception as e:
            raise CustomException(e, sys)

    

        
    
    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            dataframe = self.export_data_into_feature_store()

            logging.info("Got the data from mongodb")

            self.split_data_as_train_test(dataframe)

            logging.info("Performed train test split on the dataset")
            
            
            logging.info("Getting Output of data ingestion")

            data_ingestion_artifact = DataIngestionArtifact(
                raw_file_path=self.data_ingestion_config.feature_store_file_path,
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
                )
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            
            
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
        
        
if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise CustomException(e, sys)