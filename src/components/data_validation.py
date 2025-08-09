import os
import sys
import shutil
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.constants import SCHEMA_FILE_PATH, VALIDATION_OUTPUT_PATH, DATA_VALIDATION_VALIDATED_PATH
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, write_yaml_file



class DataValidation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        expected_columns = len(self._schema_config["columns"])
        actual_columns = len(dataframe.columns)
        logging.info(f"Expected columns: {expected_columns}, Found: {actual_columns}")
        return expected_columns == actual_columns
    
    
    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        numerical_columns = self._schema_config["numerical_columns"]
        missing = [col for col in numerical_columns if col not in dataframe.columns]
        if missing:
            logging.warning(f"Missing numerical columns: {missing}")
            return False
        return True
    
    
    
    def is_categorical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        categorical_columns = self._schema_config["categorical_columns"]
        missing = [col for col in categorical_columns if col not in dataframe.columns]
        if missing:
            logging.warning(f"Missing categorical columns: {missing}")
            return False
        return True
    
    
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)
    
    
    
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        """
        Detects dataset drift using Kolmogorov-Smirnov test.
        base_df: DataFrame of the base dataset
        current_df: DataFrame of the current dataset
        threshold: p-value threshold for drift detection
        return: True if no drift, False if drift detected
        """
        status = True
        report = {}

        for column in base_df.columns:
            d1 = base_df[column].dropna()
            d2 = current_df[column].dropna()

            if d1.dtype == "object" or d2.dtype == "object":
                le = LabelEncoder()
                combined = pd.concat([d1.astype(str), d2.astype(str)])
                le.fit(combined)
                d1 = le.transform(d1.astype(str))
                d2 = le.transform(d2.astype(str))

            _, p_value = ks_2samp(d1, d2)
            drift_status = p_value < threshold
            report[column] = {"p_value": float(p_value), "drift_status": drift_status}

            if drift_status:
                status = False

        write_yaml_file(self.data_validation_config.drift_report_file_path, content=report)
        return status
    
    
    
    def detect_prior_probability_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, target_col: str):
        """
        Detect prior probability drift for a target column.

        Args:
            base_df (pd.DataFrame): Base (reference) dataset.
            current_df (pd.DataFrame): Current dataset to compare against the base dataset.
            target_col (str): Name of the target column for which to check prior probabilities.

        Returns:
            dict: A dictionary containing prior probabilities for both datasets and the absolute differences.
        """

        
        
        base_dist = base_df[target_col].value_counts(normalize=True)
        curr_dist = current_df[target_col].value_counts(normalize=True)

        all_classes = set(base_dist.index).union(curr_dist.index)
        report = {}

        for cls in all_classes:
            base_p = float(base_dist.get(cls, 0.0))
            curr_p = float(curr_dist.get(cls, 0.0))
            report[str(cls)] = {
                "base_probability": base_p,
                "current_probability": curr_p,
                "absolute_difference": abs(base_p - curr_p)
            }

        write_yaml_file(self.data_validation_config.prior_drift_report_file_path, content=report)
        return report
    
    
    def detect_concept_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, target_col: str) -> float:
        """
        Detect concept drift using a simple logistic regression model.

        Args:
            base_df (pd.DataFrame): Base (reference) dataset.
            current_df (pd.DataFrame): Current dataset to compare against the base dataset.
            target_col (str): Name of the target column for which to check concept drift.

        Returns:
            float: Accuracy of the logistic regression model when applied to the current dataset.
        """
        
        if isinstance(target_col, list):
            target_col = target_col[0]

        le = LabelEncoder()
        base_df[target_col] = le.fit_transform(base_df[target_col])
        current_df[target_col] = le.transform(current_df[target_col])

        X_train = base_df.drop(columns=[target_col])
        y_train = base_df[target_col]
        X_test = current_df.drop(columns=[target_col])
        y_test = current_df[target_col]

        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                enc = LabelEncoder()
                all_vals = pd.concat([X_train[col], X_test[col]]).astype(str)
                enc.fit(all_vals)
                X_train[col] = enc.transform(X_train[col].astype(str))
                X_test[col] = enc.transform(X_test[col].astype(str))

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        model = LogisticRegression(max_iter=200000, solver="saga")
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, predictions)
        return float(acc)



    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiates the data validation process by reading the training and testing datasets,
        validating their schema, detecting dataset drift, prior probability drift, and concept drift,
        and saving the validated datasets and reports.

        Returns:
            DataValidationArtifact: An artifact containing the results of the data validation process.
        """
        
        try:
            train_path = self.data_ingestion_artifact.trained_file_path
            test_path = self.data_ingestion_artifact.test_file_path

            train_df = self.read_data(train_path)
            test_df = self.read_data(test_path)

            # Validate schema
            schema_ok = all([
                self.validate_number_of_columns(train_df),
                self.validate_number_of_columns(test_df),
                self.is_numerical_column_exist(train_df),
                self.is_numerical_column_exist(test_df),
                self.is_categorical_column_exist(train_df),
                self.is_categorical_column_exist(test_df)
            ])

            if not schema_ok:
                logging.error(" Schema validation failed.")

            # Drift
            drift_ok = self.detect_dataset_drift(
                base_df=train_df.drop(columns=[self._schema_config["target_column"][0]]),
                current_df=test_df.drop(columns=[self._schema_config["target_column"][0]])
            )

            # Prior prob
            _ = self.detect_prior_probability_drift(train_df, test_df, self._schema_config["target_column"][0])

            # Concept drift
            concept_acc = self.detect_concept_drift(train_df.copy(), test_df.copy(), self._schema_config["target_column"])
            concept_ok = concept_acc >= 0.7

            if not concept_ok:
                logging.warning(" Concept drift detected. Accuracy = %.2f", concept_acc)

            validation_status = schema_ok and drift_ok and concept_ok

            # Copy validated files
            validated_dir = DATA_VALIDATION_VALIDATED_PATH
            os.makedirs(validated_dir, exist_ok=True)

            validated_train_path = os.path.join(validated_dir, "train.csv")
            validated_test_path = os.path.join(validated_dir, "test.csv")

            shutil.copy(train_path, validated_train_path)
            shutil.copy(test_path, validated_test_path)

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=validated_train_path,
                valid_test_file_path=validated_test_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            write_yaml_file(file_path=VALIDATION_OUTPUT_PATH, content=artifact.__dict__)
            logging.info(f" Data Validation Artifact: ===========>>>>>{artifact}<==============")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion_config = DataIngestionConfig()
    data_ingestion_artifact = DataIngestionArtifact(
        raw_file_path=data_ingestion_config.feature_store_file_path,
        trained_file_path=data_ingestion_config.training_file_path,
        test_file_path=data_ingestion_config.testing_file_path
    )

    data_validation_config = DataValidationConfig()

    validation = DataValidation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validation_config=data_validation_config
    )
    validation.initiate_data_validation()
