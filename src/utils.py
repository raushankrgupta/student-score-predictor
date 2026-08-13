import os
import sys
import pickle
from src.exception import CustomException
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV





def save_object(file_path,file_obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(file_obj,file=file)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,Y_train,X_test,Y_test,models,params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            param = params[model_name]
            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,Y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,Y_train)

            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)

            train_model_score = r2_score(Y_train,Y_train_pred)
            test_model_score = r2_score(Y_test,Y_test_pred)

            report[model_name] = test_model_score
            
        return report


    except Exception as e:
        raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
