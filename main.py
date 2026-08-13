from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__=='__main__':
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()
    logging.info(f"Train Path: {train_path}")
    logging.info(f"Test Path: {test_path}")
    transformer = DataTransformation()
    train_arr,test_arr,_ = transformer.initiate_data_transformation(train_path,test_path)
    logging.info(f"Train Array Shape: {train_arr.shape}")
    logging.info(f"Test Array Shape: {test_arr.shape}")
    trainer = ModelTrainer()
    r2Score = trainer.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr)
    logging.info(f"Final R2 Score: {r2Score}")


