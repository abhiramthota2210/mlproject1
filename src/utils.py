import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from logger import logging


from excepetion import CustomExcepetion

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb")as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomExcepetion(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_score = float("-inf")
        best_model = None
        best_model_name = None

        for name, model in models.items():
            para = params.get(name, {})

            if para:
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)
                tuned_model = gs.best_estimator_
                logging.info(f"GridSearch used for {name}")
            else:
                model.fit(X_train, y_train)
                tuned_model = model
                logging.info(f"No GridSearch for {name}")

            y_test_pred = tuned_model.predict(X_test)
            score = r2_score(y_test, y_test_pred)

            report[name] = score

            if score > best_score:
                best_score = score
                best_model = tuned_model
                best_model_name = name

        logging.info(f"Best model: {best_model_name} with R2: {best_score}")

        return report, best_model_name

    except Exception as e:
        raise CustomExcepetion(e, sys)

def load_object(file_path):
    try:
        with open(file_path,"rb")as file_obj:   
            return dill.load(file_obj) 
            
    except Exception as e:
        raise CustomExcepetion(e,sys)