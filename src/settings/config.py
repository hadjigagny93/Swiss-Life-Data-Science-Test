import os
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DATA_DIR = os.path.join(REPO_DIR, 'data')
LOGS_DIR = os.path.join(REPO_DIR, 'logs')
MODELS_DIR = os.path.join(REPO_DIR, 'src/application/models/')


DEFAULT_TRAIN_SIZE =  14900


UNUSEFUL_FEATURES = 'cheveux'
EXP_LEVEL = 'exp_level'
EXP = 'exp'
DATE = 'date'
AGE = 'age'
SALAIRE = 'salaire'
PROFIL = 'profil'
TARGET = 'target'
NUM_FEATURES = ['age', 'exp', 'salaire', 'note', 'day_date', 'week_date', 'month_date', 'weekday_date']
CAT_FEATURES = ['sexe', 'diplome', 'specialite', 'dispo', 'exp_level', 'profil']


fit_params = {
    'early_stopping_rounds': 50, 
    'eval_metric' : 'auc', 
    'eval_names': ['valid'],
    'verbose': 100,
    'categorical_feature': 'auto'}

param_test = {
    'num_leaves': sp_randint(6, 50), 
    'min_child_samples': sp_randint(100, 500), 
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'subsample': sp_uniform(loc=0.2, scale=0.8), 
    'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
    }
