import sys
import os
from dataclasses import dataclass
from src import constants
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(constants.DATA_PATH,constants.TRAIN_DATA_FILENAME)
    test_data_path: str = os.path.join(constants.DATA_PATH,constants.TEST_DATA_FILENAME)
    raw_data_path: str = os.path.join(constants.DATA_PATH,constants.RAW_DATA_FILENAME)


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting Data Ingestion")
        try:
            df = pd.read_csv(constants.SOURCE_DATA_PATH)
            logging.info("Read source data completed.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,header=True,index=False)
            logging.info("Initiating train test Split.")

            train_set, test_set = train_test_split(df,test_size=0.2,random_state=45)

            train_set.to_csv(self.ingestion_config.train_data_path,header=True,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,header=True,index=False)

            logging.info("Data ingestion is complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
