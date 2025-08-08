import sys

from src.exception import CustomException
from src.logger import logging

import os
from src.constants import DB_NAME
import pymongo
import certifi
from dotenv import load_dotenv


load_dotenv()

# Load the CA certificate for secure MongoDB connection
ca = certifi.where()

class MongoDBClient:
    """
    Class Name :   export_data_into_feature_store
    Description :   This method exports the dataframe from mongodb feature store as dataframe 
    
    Output      :   connection to mongodb database
    On Failure  :   raises an exception
    """
    client = None

    def __init__(self, database_name=DB_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv("MONGODB_URL")
                if mongo_db_url is None:
                    raise Exception(f"Environment key is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection succesfull")
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__ == "__main__":
    mongo_client = MongoDBClient()
    print(mongo_client.database_name)
    print("MongoDB connection established successfully.")