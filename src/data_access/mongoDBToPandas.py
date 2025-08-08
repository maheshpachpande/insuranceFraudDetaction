from src.configuration.mongoDB_connection import MongoDBClient
from src.constants import DB_NAME, COLLECTION_NAME
from src.exception import CustomException
import pandas as pd
import sys
from typing import Optional
import numpy as np



class InsuranceData:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self):
        """
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DB_NAME)
        except Exception as e:
            raise CustomException(e,sys)
        

    def export_collection_as_dataframe(self,collection_name:str,database_name:Optional[str]=None)->pd.DataFrame:
        try:
            """
            export entire collectin as dataframe:
            return pd.DataFrame of collection
            """
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
                print(collection)
            else:
                collection = self.mongo_client[database_name][collection_name]
                print(collection)

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    insurance_data = InsuranceData()
    df = insurance_data.export_collection_as_dataframe(collection_name=COLLECTION_NAME)
    print(df.head())
    print("Data exported successfully as DataFrame.")