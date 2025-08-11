import sys
import numpy as np
import pandas as pd

from imblearn.combine import SMOTETomek

from src.logger import logging
from src.exception import CustomException
from src.target_mapping import TargetValueMapping
from src.feature_engineering import DataTransformer
from src.constants import TARGET_COLUMN, VALIDATION_OUTPUT_PATH
from src.utils.main_utils import save_numpy_array_data, save_object, read_yaml_file
from src.entity.config_entity import DataValidationConfig, DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact



class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self.data_transformer = DataTransformer()
        except Exception as e:
            raise CustomException(e, sys)
        
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
        
        
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                # raise CustomException("Data Validation failed.")
                logging.warning("⚠️ Data validation failed. Proceeding anyway (debug mode).")
            
            # if not data_validation_artifact.validation_status:
            #     logging.warning("⚠️ Data validation failed. Proceeding anyway (debug mode).")

            logging.info("Getting data from data validation artifact. Reading train and test data.")
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            logging.info("Input feature into train and test dataframes with target mapping.")
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

            target_feature_train_df = train_df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())
            target_feature_test_df = test_df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())

            logging.info("Creating preprocessor object for data transformation.")
            preprocessor = self.data_transformer.get_data_transformer_object(input_feature_train_df)
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            
            print(input_feature_train_df.columns)

            logging.info("Transforming input features for train and test dataframes.")
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            logging.info("Applying SMOTETomek for handling class imbalance.")
            smt = SMOTETomek(sampling_strategy="minority")
            train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df)
            
            input_feature_train_final = train_final[0]
            target_feature_train_final = train_final[1]
            
            test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df)
            
            input_feature_test_final = test_final[0]
            target_feature_test_final = test_final[1]

            logging.info("Saving transformed data as numpy arrays.")
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    
    data_validation_config = DataValidationConfig()
    

    cnf = read_yaml_file(VALIDATION_OUTPUT_PATH)
    
    data_validation_artifacts = DataValidationArtifact(
        validation_status=cnf.validation_status,
        valid_train_file_path=cnf.valid_train_file_path,
        valid_test_file_path=cnf.valid_test_file_path,    
        drift_report_file_path=cnf.drift_report_file_path
    )
    
    data_transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation(data_validation_artifacts, data_transformation_config)
    data_transformation.initiate_data_transformation()