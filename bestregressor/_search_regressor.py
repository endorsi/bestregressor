from datetime import timedelta
from sklearn.model_selection import train_test_split, RandomizedSearchCV, PredefinedSplit, GridSearchCV
import sklearn.metrics as smetric
from sklearn.pipeline import Pipeline        
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,ElasticNet,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from abc import ABCMeta,abstractmethod 
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from pathlib import Path


class BaseBestRegressor(metaclass=ABCMeta):
    """ Searchs for the best regressor with hyperparameter optimization """
    
    @abstractmethod
    def __init__(self, *, lags = [], 
                 val_ratio = 0.1, test_ratio = 0.1):
        
        self.lags = lags
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self._check_base()
        
        self.__searchMethodisCalled = False
        self.defined_models = [SVR(), RandomForestRegressor(), Lasso(), ElasticNet(), Ridge()] 

    def __str__(self):
        raise NotImplementedError("Must Override __str__")
    
    def filter_defined_models(self, index):
        
        if not isinstance(index, list):
            raise TypeError("The input must be a list to filter")
        
        if not all(isinstance(x, int) and x > -1 for x in index):
            raise ValueError("The input must only include non-negative integers")
            
        self.defined_models = [self.defined_models[i] for i in index]
        
    def _json_converter(self, json_file):
        
        model_params = json.load(json_file)
        model_params_range = self._convert_ranges(model_params)
        models_dict = self._convert_models(model_params_range)
        
        return models_dict
    
    def _convert_ranges(self, model_params):
        
        for model_name in model_params.keys():
            for parameter, values in model_params[model_name].items():
                try:
                    if "low" in values.keys():
                        low, high, step = values["low"], values["high"], values["step"]
                        model_params[model_name][parameter] = np.arange(low, high, step)
                except:
                    pass
                
        return model_params
    
    def _convert_models(self, model_params):
        
        models_dict = {}
        for model_name in self.defined_models:
            try:
                models_dict[model_name] = model_params[str(model_name)[:-2]]
            except: 
                pass
            
        return models_dict
    
    def _check_base(self):
        
        if not isinstance(self.lags, list):
            raise TypeError("lags must be a list")
        
        if not all(isinstance(x, int) and x > 0 for x in self.lags):
            raise ValueError("lags must only include positive integers")
            
        if not self.val_ratio > 0 and self.test_ratio > 0 :
            raise ValueError("val_ratio and test_ratio must be a positive number")
        
        if not self.val_ratio + self.test_ratio <= 0.99:
            raise ValueError("val_ratio + test ratio must be equal or less than 0.99")
    
    def _check_search(self, df, target_name):
        
        # scikit-learn checks that all the data are numerical, non-null, not infinite itself
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        
        if not isinstance(target_name, str):
            raise TypeError("target_name must be a string that represents y column")
        
        if not target_name in df.columns:
            raise KeyError("target_name must be a column name in df")
            
        train_length = len(df) * ( 1 - (self.test_ratio + self.val_ratio) )
        
        if not 50 <= train_length <= 100000:
            raise Exception("The total number of training samples must be between 50 and 100K")
        
        if not len(df.columns) <= 100000:
            raise Exception("The total number of features must be lower than 100K")
            
        if self.lags:
            if not max(self.lags) < train_length:
                raise ValueError("The maximum lags number must be lower than the total number of training samples")
                                 
        if not self.lags:
            if not len(df.columns) >= 2:
                raise Exception("There must be at least two features if the lags list is empty")
    
    def _create_new_columns(self, lags):

        new_column_names = [str(lag) + "Before" for lag in lags]
        return new_column_names
        
    def _add_previous_values(self, df, target_name, lags, new_column_names):
        
        # adds the previous values of y_column as a feature
        for lag,new_column_name in zip(lags, new_column_names):
            df[new_column_name] = df[target_name].shift(lag,fill_value = df[target_name][:max(lags)].median())
        
        return df
    
    def split_X_y(self, df, target_name):
        
        X = df.drop(target_name,axis=1)   
        y = df[target_name]
        
        return [X, y]
    
    def _get_model_params(self):
        raise NotImplementedError("Must Override model_params")
    
    def _search_type(self):
        raise NotImplementedError("Must Override search_type")
    
    def _get_preds(self, X_test, trained):
        
        y_preds = []
                
        for index in X_test.index:
            pred = trained.predict(X_test.loc[index,:].values.reshape(1,-1))
            pass
            y_preds.append(pred[0])
            
            if self.lags:
                # writes the prediction to the further features as a previous value 
                for lag,column_name in zip(self.lags, self.new_column_names):
                    new_index = index + timedelta(lag)
                    if new_index in X_test.index:
                        X_test.loc[new_index,column_name] = pred
        
        return y_preds
    
    
    def search(self, df, target_name):
        """df is a pandas DataFrame, target_name is a string that represents y-column"""
        self._check_search(df, target_name)
        
        warnings.filterwarnings('ignore')
        
        if self.lags:
            self.new_column_names = self._create_new_columns(self.lags)
            df = self._add_previous_values(df, target_name, self.lags, self.new_column_names)
        
        self.__X, self.__y = self.split_X_y(df, target_name)
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(self.__X, self.__y, 
                                                   test_size = self.test_ratio, shuffle = False)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, 
                                         test_size = self.val_ratio/(1-self.test_ratio), shuffle=False)
        
        # look at : https://scikit-learn.org/stable/modules/cross_validation.html#predefined-fold-splits-validation-sets
        split_index = [-1 if x in X_train.index else 0 for x in X_train_val.index]
        
        ps = PredefinedSplit(test_fold = split_index)
        
        self.models_dict = self._get_model_params()
        
        self.model_val_scores = {}
        self.model_test_scores = {}
        self.model_parameters = {}
        self.model_values = {}
        self.trained_models = {}
        self.untrained_models = {}

        for model,params in self.models_dict.items(): 
            
            model_name = str(model)[:-2]
            scale = params.pop('scale_data', False)
            
            if scale:
                params = {model_name + "__" + k : v for k,v in params.items()}
                model = Pipeline([('scaler', StandardScaler()), (model_name, model)])
        
            search = self._search_type(model, params, ps)
            trained = search.fit(X_train_val, y_train_val)
            
            y_preds = self._get_preds(X_test, trained)

            r2_sc = smetric.r2_score(y_test,y_preds)   

            self.model_val_scores[model_name] = str(trained.best_score_)                                  
            self.model_test_scores[model_name] = r2_sc
            self.model_parameters[model_name] = trained.best_params_ 
            self.model_values[model_name] = {"Predictions" : y_preds.copy(), "Test Series" : y_test}
            self.trained_models[model_name] = trained
            self.untrained_models[model_name] = model
        
        
        self.__searchMethodisCalled = True
    
    def fit_best(self):
        
        if not self.__searchMethodisCalled:
            raise Exception("The search method must be called first to get the best regressor.") 
            
        scores = self.model_test_scores

        best_model_name = max(scores, key = scores.get)
        
        best_model_params = self.model_parameters[best_model_name]
        best_model = self.untrained_models[best_model_name]
        
        if type(best_model) == Pipeline:
            best_model_params = { k.split("__")[1] : v for k,v in best_model_params.items()}
            best_model[1].set_params(**best_model_params)
        else:
            best_model.set_params(**best_model_params)
        
        X_train, X_test, y_train, y_test = train_test_split(self.__X, self.__y, 
                                           test_size = self.test_ratio, shuffle = False)
        
        trained = best_model.fit(X_train, y_train)
        
        y_preds = self._get_preds(X_test, trained)
        
        self.best_model_values = {"Predictions" : y_preds.copy(), "Test Series" : y_test}
        self.best_model_trained = trained
        
        return trained
    
    def _plot(self, test_preds, legends, *, resample=False, abb=None, indexing=False, index1=None, index2=None):
    
        if resample:
            x_all = [y.resample(abb).mean() for y in test_preds]
        elif indexing:
            x_all = [y[index1:index2] for y in test_preds]
        else:
            x_all = test_preds

        fig,ax = plt.subplots(figsize=(15,7))
        
        for x in x_all:
            ax.plot(x, marker = "o")

        ax.set_title("Truth-Prediction(s) Comparison")

        ax.legend(legends, loc='upper right')
        
        return fig
        
    def plot_models(self, resample=False, abb="Y", indexing=False, index1=None, index2=None, plot_best=False):
        
        if not self.__searchMethodisCalled:
            raise Exception("The search method must be called first to plot model results.") 
        
        sns.set_theme(style="whitegrid")

        if plot_best:
            scores = self.model_test_scores
            best_model_name = max(scores, key = scores.get)
            model_values = self.model_values[best_model_name]

            test = [model_values["Test Series"]]
            preds = [pd.Series(data = model_values["Predictions"], index = test[0].index)]
            legends = ["Truth" , "Best Model : " + best_model_name]
        
        else:
            test = [next(iter(self.model_values.values()))["Test Series"]]
            preds = [pd.Series(data = values["Predictions"], index = test[0].index) for values in self.model_values.values()]    
            legends = ["Truth"] + list(self.model_values.keys())
        
        test_preds = test + preds

        fig = self._plot(test_preds, legends, resample=resample, abb=abb, 
                         indexing=indexing, index1=index1, index2=index2)
        
        return fig
    
class RandomizedBestRegressor(BaseBestRegressor):
    """ Searchs for the best regressor with randomized search hyperparameter optimization """
    
    def __init__(self, *, lags = [], 
                 val_ratio = 0.1, test_ratio = 0.1, n_iter = 20):
        
        self.n_iter = n_iter
        self._check_randomized()

        super().__init__(
                lags = lags, val_ratio = val_ratio,
                test_ratio = test_ratio) 
    
    def _check_randomized(self):
        
        if not 1 <= self.n_iter <= 500:  
            raise ValueError("n_iter must be a numerical value between 1 and 500")
    
    def _get_model_params(self):
        
        file_location = Path(__file__).absolute().parent
        json_file = open(file_location / 'random_model_params.json')
        
        self.__models_dict_random = super()._json_converter(json_file)
        return self.__models_dict_random
    
    def _search_type(self, model, params, ps):
        
        search = RandomizedSearchCV(model, params, cv = ps, n_iter = self.n_iter, scoring = "r2", refit=True)     
        return search
    
    def __str__(self):
        
        info = (f"lags = {self.lags}, validation ratio = {self.val_ratio}, " 
        f"test ratio = {self.test_ratio}, n_iter = {self.n_iter}")
        return info
            
class GridBestRegressor(BaseBestRegressor):
    """ Searchs for the best regressor with grid search hyperparameter optimization """
    
    def __init__(self, *, lags = [], 
                 val_ratio = 0.1, test_ratio = 0.1, n_jobs = -1):
        
        self.n_jobs = n_jobs
        self._check_grid()
        
        super().__init__(
                lags = lags, val_ratio = val_ratio,
                test_ratio = test_ratio) 
    
    def _check_grid(self):
        
        if not self.n_jobs in {None, -1}: 
            raise ValueError("n_jobs must be None or -1")
                
    def _get_model_params(self):
        
        file_location = Path(__file__).absolute().parent
        json_file = open(file_location / 'grid_model_params.json')
        
        self.__models_dict_grid = super()._json_converter(json_file)
        return self.__models_dict_grid
    
    def _search_type(self, model, params, ps):
        
        search = GridSearchCV(model, params, cv = ps, n_jobs = self.n_jobs, scoring = "r2", refit=True)   
        return search
    
    def __str__(self):
        
        info = (f"lags = {self.lags}, validation ratio = {self.val_ratio}, " 
        f"test ratio = {self.test_ratio}, n_jobs = {self.n_jobs}")
        return info