# import modules
import os
import pandas as pd
import numpy as np
import src.settings.config as config
from src.infrastructure.utils import exp__level, profil, create_date_features
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn_pandas import CategoricalImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer


class  ModelDiagnostic:
    """class model that get the whole info about the pipelines
    steps which have already been performed and the others to come
    """
    pass 


class RemoveOutliers(BaseEstimator, TransformerMixin):
    """ remove outliers and keep index of interest for label
    
    attributes:
    -----------
    indkeep: class attr -- register unremoved index
    job: train or test -- do not remove outliers during test pipeline
    """
    
    indkeep = {}

    def __init__(self, job="train"):
        self.job = job

    def fit(self, X):
        return self


    @classmethod
    def remove_outliers_memory(cls, arg):
        cls.indkeep['u'] = arg

    def transform(self, X):
        "Do not touch anything if job equals test"
        if self.job == "test":
            return X
        for var in config.NUM_FEATURES:
            X = X[np.abs(X[var] - X[var].mean()) <= (3 * X[var].std())]
        cls = self.__class__
        # register index
        cls.remove_outliers_memory(arg=X.index)
        return X

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """ select features depending on which pipeline is running
    attributes:
    -----------
    name_of_attributes: same paradigm as indkeep (RemoveOutliers class)
    typ: numeric or categorical data
    """
    name_of_dataframe = {}

    def __init__(self, features, typ="num"):
        self.features = features
        self.typ = typ

    def fit(self, X):
        return self

    @classmethod 
    def dataframe_selector_memory(cls, arg):
        key, col = arg
        cls.name_of_dataframe[key] = col


    def transform(self, X):
        col = list(set(X.columns.tolist()).intersection(self.features))
        cls = self.__class__
        cls.dataframe_selector_memory((self.typ, col))
        if self.typ == "num":
            return X[col]
        return X[col]

class RemoveFeatures(BaseEstimator, TransformerMixin):
    """remove some features -- in this case only cheveux_col will be removed"""
    def fit(self, X):
        return self

    def transform(self, X):
        return X.drop(config.UNUSEFUL_FEATURES, axis=1)

class CreateNewFeatures(BaseEstimator, TransformerMixin):
    """create new features for feature engineering """
    def fit(self, X):
        return self

    def transform(self, X):
        X[config.EXP_LEVEL] = X[config.EXP].apply(exp__level)
        X[config.PROFIL] = X[config.AGE].apply(profil)
        X = create_date_features(X, config.DATE)
        return X

class RebuildDataFrame(BaseEstimator, TransformerMixin):
    """ Artifact for conserving dataframe column names needed in features importance"""

    def fit(self, X):
        return self

    def transform(self, X):
        categorical_imputer_matrix_name = DataFrameSelector.name_of_dataframe['cat']
        return pd.DataFrame(X, columns=categorical_imputer_matrix_name)


class CustomCategoricalImputer(BaseEstimator, TransformerMixin):
    """categorical imputer"""
    def fit(self, X):
        return self

    def transform(self, X):
        for var in config.CAT_FEATURES:
            imputer = CategoricalImputer()
            X[var] = imputer.fit_transform(X[var])
        return X 

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """ label encoder """
    def fit(self, X):
        return self

    def transform(self, X):
        for var in config.CAT_FEATURES:
            X[var] = LabelEncoder().fit_transform(X[var].values)
        return X 
  
class GetIndexToKeep(BaseEstimator, TransformerMixin):
    """ reuse remove outliers class attributes to return only remained index 
    after removing outliers on features data, it will be used by target series
    """
    def fit(self, X):
        return self

    def transform(self, X):
        return X, RemoveOutliers.indkeep

class LogTransform(BaseEstimator, TransformerMixin):
    """log transform for (or pseudo) countable data """
    def fit(self, X):
        return self

    def transform(self, X):
        X[config.AGE] = FunctionTransformer(np.log1p).transform(X[config.AGE])
        X[config.EXP] = FunctionTransformer(np.log1p).transform(X[config.EXP])
        X[config.SALAIRE] = FunctionTransformer(np.log1p).transform(X[config.SALAIRE])
        return X

class Process:
    """data processing and feature engineering
    attributes:
    -----------
    job: str -- train or test 
    
    methods:
    --------
    build_pipelines: create all process and feature engineering pipelines 
    get_data: return processed data $
    """

    def __init__(self, job="train"):
        self.job = job

    def build_pipelines(self):
        process_pipeline = Pipeline(steps=[
            ("feature_creation", CreateNewFeatures()),
            ("remove_unuseful_features", RemoveFeatures()),
            ("remove_outliers", RemoveOutliers(job=self.job)),
            ])
        numeric_pipeline = Pipeline(steps=[
           ("selector", DataFrameSelector(config.NUM_FEATURES, typ="num")),
           ("log_transform", LogTransform()),
           ("imputer", SimpleImputer(
            strategy="median"
           ))
           ]) 
        
        categorical_pipeline = Pipeline(steps=[
            ("selector", DataFrameSelector(config.CAT_FEATURES, typ="cat")),
            ("categorical_imputer", CustomCategoricalImputer()),
            ("rebuild_dataframe", RebuildDataFrame()),
            ("labelencoder", CustomLabelEncoder())])

        num_and_cat_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", numeric_pipeline),
            ("cat_pipeline", categorical_pipeline)])


        full_common_pipeline = Pipeline(steps=[
           ("process_pipeline", process_pipeline),
           ("num_and_cat_pipeline", num_and_cat_pipeline),
           ("get_index", GetIndexToKeep())])
     
        return full_common_pipeline

    def get_data(self, df):
        
        features = df.drop('target', axis=1)
        target = df.target
        pipeline = self.build_pipelines()
        new_data, idx = pipeline.fit_transform(features)
        dataframe_columns = DataFrameSelector.name_of_dataframe['num'] + DataFrameSelector.name_of_dataframe['cat']
        new_features = pd.DataFrame(new_data, columns=dataframe_columns)
        if self.job == 'train':
            new_target = target[idx['u']].reset_index(drop=True)
        else:
            new_target = target
        return new_features,  new_target