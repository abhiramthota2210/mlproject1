import sys 
from dataclasses import dataclass
 
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from excepetion import CustomExcepetion
from logger import logging
from sklearn.metrics import r2_score
from utils import save_object
from utils import evaluate_models



@dataclass
class ModelTrainerConfig:
    trainer_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
        
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Classifier" : KNeighborsRegressor(),
                "XGBClassifier" : XGBRegressor(),
                "CatBoosting Classifier":CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
                
            }
            
            #get the best model score from the dict
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                              models=models)
            #get best model ame from dict
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

           
            best_model=models[best_model_name]
            
            
            if best_model_score<0.6:
                raise CustomExcepetion("No best model found")
            
            logging .info(f"Best model on both training and testing dataset")
            
            best_model.fit(X_train,y_train)

            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            #print(r2_square)
            return r2_square
            
            
            
        except Exception as e:
            
            raise CustomExcepetion(e,sys)