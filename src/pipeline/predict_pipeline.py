import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src import constants
from src.utils import load_object


class PredictionPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            logging.info("Getting model and preprocessor file paths")
            model_path = os.path.join(constants.DATA_PATH,constants.TRAINED_MONDEL_FILENAME)
            preprocessor_path = os.path.join(constants.DATA_PATH,constants.PREPROCESSOR_FILENAME)

            logging.info("Loading model and preprocessor file paths")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("Predicting score")
            scaled_data = preprocessor.transform(features)
            pred = model.predict(scaled_data)
            return pred
        except Exception as e:
            raise CustomException(e,sys)




class CustomData:
    def __init__(
            self,
            gender: str,
            race_ethnicity: str,
            parental_level_of_education: str,
            lunch: str,
            test_preparation_course: str,
            reading_score: int,
            writing_score: int
        ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)