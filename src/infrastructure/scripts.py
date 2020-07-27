# librairies import 
import os
import datetime
import numpy as np
import pandas as pd 
from src.settings.config import DATA_DIR, DEFAULT_TRAIN_SIZE

# this script enables creating train and test files by 
# read csv from data repository
# remove index columns 
# remove rows with nan values on date columns
# convert string date format to datetime one for sort dataframe 
# reset index finally
def get_date_nan_value_out_of_there(x):
    return x is np.nan 

data = pd.read_csv(os.path.join(DATA_DIR, 'data.csv'), sep=',')
data = data \
    .drop(['Unnamed: 0', 'index'], axis=1) \
    .drop(data[data.date.apply(get_date_nan_value_out_of_there) == True].index) \
    .assign(date=lambda df: df.date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))) \
    .sort_values(by='date') \
    .reset_index(drop=True) 
    
# rename embauche
data.rename(columns={'embauche': 'target'}, inplace=True)

# create train and test csv files 
data.iloc[:DEFAULT_TRAIN_SIZE,:].to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
data.iloc[DEFAULT_TRAIN_SIZE:,:].to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)
