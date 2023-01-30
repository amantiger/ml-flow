#!/usr/bin/env python
# coding: utf-8

# In[3]:


# importing modules 
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy import stats 
from scipy.stats import expon, reciprocal
import mlflow
import mlflow.sklearn

remote_server_uri = "http://0.0.0.0:5000" # set to the server URI
mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

exp_name = "Housing_value_prediction_mlflow"
mlflow.set_experiment(exp_name)

#Loading the datasets

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def model():
    with mlflow.start_run(run_name="PARENT RUN",nested=True) as parent_run:
    

        def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
            if not os.path.isdir(housing_path):
                os.makedirs(housing_path)
            tgz_path = os.path.join(housing_path, "housing.tgz")
            urllib.request.urlretrieve(housing_url, tgz_path)
            housing_tgz = tarfile.open(tgz_path)
            housing_tgz.extractall(path=housing_path)
            housing_tgz.close()

#loading the dataset

        def load_housing_data(housing_path=HOUSING_PATH):
            csv_path = os.path.join(housing_path, "housing.csv")
            return pd.read_csv(csv_path)

#startified splitting

        def strat_split():
            fetch_data = fetch_housing_data()
            housing = load_housing_data()
            housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
            housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    
            split = StratifiedShuffleSplit(n_splits=1,test_size=0.2 ,random_state=42)
            for train_index, test_index in split.split(housing,housing["income_cat"]):
                strat_train_set = housing.loc[train_index]
                strat_test_set = housing.loc[test_index]
            # droping income_cat column from the splitted datasets to return to original form
            for set in (strat_train_set, strat_test_set):
                set.drop(["income_cat"], axis=1, inplace=True)
            return strat_train_set,strat_test_set

        #seprating target variables and predictors
        def label_seprate():
            strat_train_set,strat_test_set = strat_split()
            housing = strat_train_set.drop("median_house_value", axis=1)
            housing_labels = strat_train_set["median_house_value"].copy()
            return housing,housing_labels

        housing,housing_labels = label_seprate()

        def dtype_seprate():
            # getting the numerical and categorical columns
    
            #housing,housing_labels = label_seprate()

            num_cols = housing.select_dtypes(include=np.number).columns
            cat_cols = housing.select_dtypes(exclude=np.number).columns
    
            return num_cols,cat_cols

        #Generating attributes

        col_names = "total_rooms", "total_bedrooms", "population", "households"
        rooms_ix, bedrooms_ix, population_ix, households_ix = [housing.columns.get_loc(c) for c in col_names]

        class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
                def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
                    self.add_bedrooms_per_room = add_bedrooms_per_room
                def fit(self, X, y=None):
                    return self  # nothing else to do
                def transform(self, X):
                    rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
                    population_per_household = X[:, population_ix] / X[:, households_ix]
                    if self.add_bedrooms_per_room:
                        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                        return np.c_[X, rooms_per_household, population_per_household,
                                 bedrooms_per_room]
                    else:
                        return np.c_[X, rooms_per_household, population_per_household]

        #Pipeline

        def pipe():
            with mlflow.start_run(run_name = "pipline",nested=True):
        
                num_cols,cat_cols = dtype_seprate()
        
                num_pipeline = Pipeline(
                [("imputer",SimpleImputer(strategy="median")),
                 ("attribs_adder",CombinedAttributesAdder()),
                 ("std_scaler", StandardScaler())
                ])
    
                num_attribs = list(housing[num_cols])
                cat_attribs = ["ocean_proximity"]
    
                full_pipeline = ColumnTransformer(
                [ ("num",num_pipeline,num_attribs),
                  ("cat",OneHotEncoder(),cat_attribs),
                ])
        
                housing_prepared = full_pipeline.fit_transform(housing)
        
                mlflow.log_param(key="imputer",value=full_pipeline.transformers_[0][1].named_steps["imputer"].get_params())
                mlflow.log_param(key="custom_transformer",value=full_pipeline.transformers_[0][1].named_steps["attribs_adder"].get_params())
                mlflow.log_param(key="Standardiser",value=full_pipeline.transformers_[0][1].named_steps["std_scaler"].get_params())
                mlflow.log_param(key="OneHotEncoder",value=full_pipeline.transformers_[1][1].get_params())
                print("Save to: {}".format(mlflow.get_artifact_uri()))
                mlflow.end_run()
        
                return housing_prepared
        
        housing_prepared=pipe()
        
        def eval_metrics(actual, pred):
            # compute relevant metrics
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            
            return rmse, mae
        
        #Linear Regressor func
        def lin_reg():
            with mlflow.start_run(run_name="LinearReg",nested=True):
        
                lin_reg = LinearRegression()
    
                lin_reg.fit(housing_prepared, housing_labels)
                housing_predictions = lin_reg.predict(housing_prepared)
        
                (rmse,mae)=eval_metrics(housing_labels,housing_predictions)
        
                # Log parameter, metrics, and model to MLflow
                mlflow.log_metric(key="rmse", value=rmse)
                mlflow.log_metrics({"mae": mae})
                return LinearRegression()
        
        lin_reg()
            
        #Tree Regressor func
        def tree_reg():
            with mlflow.start_run(run_name="DecisionTree",nested=True):
                tree_reg= DecisionTreeRegressor(random_state=42)
        
                mlflow.log_param(key="random state",value=tree_reg.random_state)
        
                tree_reg.fit(housing_prepared, housing_labels)
                housing_predictions = tree_reg.predict(housing_prepared)
        
                (rmse,mae)=eval_metrics(housing_labels,housing_predictions)
        
                #Log parameter, metrics, and model to MLflow
            
                mlflow.log_metric(key="rmse", value=rmse)
                mlflow.log_metrics({"mae": mae})
                mlflow.end_run()
                return tree_reg
            
        tree_reg()
            
        #Forest Regressor func
        
        def forest_reg():
            with mlflow.start_run(run_name="RandomForest",nested=True):
                forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        
                forest_reg.fit(housing_prepared, housing_labels)
                housing_predictions = forest_reg.predict(housing_prepared)
        
                (rmse,mae)=eval_metrics(housing_labels,housing_predictions)
        
                #Log parameter, metrics, and model to MLflow
        
                mlflow.log_metric(key="rmse", value=rmse)
                mlflow.log_metrics({"mae": mae})
                mlflow.log_params({"n_estimators":forest_reg.n_estimators,"random state":forest_reg.random_state})
                mlflow.end_run()
                return forest_reg
            
        forest_reg()
            
        #SVM regressor
        def svm_reg():
            with mlflow.start_run(run_name="SVR",nested=True):
                svm_reg= SVR(kernel="linear")
        
                svm_reg.fit(housing_prepared, housing_labels)
                housing_predictions = svm_reg.predict(housing_prepared)
        
                (rmse,mae)=eval_metrics(housing_labels,housing_predictions)
        
                #Log parameter, metrics, and model to MLflow
        
                mlflow.log_metric(key="rmse", value=rmse)
                mlflow.log_metrics({"mae": mae})
                mlflow.log_param(key="kernel",value=svm_reg.kernel)
                mlflow.end_run()
                return svm_reg
            
        svm_reg()
        
        #Cross Validation
        def cross_val(reg_model):
            with mlflow.start_run(run_name="cross_val",nested=True):
                scores = cross_val_score(reg_model, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
        
                rmse_scores = np.sqrt(-scores)
                mlflow.log_params({"scores":rmse_scores,"model":type(reg_model)})
                mlflow.log_metrics({"Mean":rmse_scores.mean(),"Standard deviation":rmse_scores.std()})
        
                mlflow.end_run()
        
                return scores
        
        cross_val(lin_reg())
        cross_val(forest_reg())
        
        #GridSearch CV
        def grid_search_cv(reg_model):
            with mlflow.start_run(run_name="grid_searvh_cv",nested=True):
        
                param_grid = [
                    # try 12 (3×4) combinations of hyperparameters
                    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                    # then try 6 (2×3) combinations with bootstrap set as False
                    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
                    ]
        
                grid_search = GridSearchCV(reg_model, param_grid, cv=5,
                                   scoring='neg_mean_squared_error',
                                   return_train_score=True)
                mlflow.log_params({"parameters":param_grid,"model":type(reg_model),"scoring":grid_search.scoring})
        
                mlflow.end_run()
                return grid_search
        
        
        
        grid_search = grid_search_cv(forest_reg())
        grid_search.fit(housing_prepared, housing_labels)
        
        def final_model(best_estimator):
            with mlflow.start_run(run_name="Final_model",nested=True):
                final_model = best_estimator
        
                final_prediction = final_model.predict(housing_prepared)
        
                (rmse,mae)=eval_metrics(housing_labels,final_prediction)
                mlflow.log_metric(key="rmse", value=rmse)
                mlflow.log_metrics({"mae": mae})
                mlflow.end_run()
                print("final_rmse: {}".format(rmse))
                print("final_mae: {}".format(mae))
                
        final_model(grid_search.best_estimator_)
        
        mlflow.end_run()
        
model()


# In[ ]:




