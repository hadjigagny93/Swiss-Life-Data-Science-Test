
import pandas as pd 
import numpy as np 
from scipy import stats
from scipy.stats import chi2
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

class HypothesisTest:
    """ general class for performing hypothesis testing over data features

    attributes:
    -----------
    df: dataframe
    info: summary info returned by test 
    feature_1: first variable
    feature_2: second variable
    
    methods:
    --------
    _get_column_type: return data dtype column (num or cat)
    _get_type_of_test: return dict -- which test to perform based on features dtypes 
    _get_decision: return summary 
    _hypothesis_chi_two: test for two cat variables
    _hypothesis_rel: test on two num variables 
    _hypothesis_oneway: test on hybrid variable, warnings !!! harcoded variable ... will not be commited
    _test: main function 
    """
    def __init__(self, df, features):
        self.df = df
        try:
            self.feature_1, self.feature_2 = features
        except ValueError:
            raise 'features must be an iterable of len 2'
        self.info = {}

    @staticmethod
    def _get_column_type(df, feature):
        return df.dtypes[feature] == np.dtype
    
    def _get_type_of_test(self):
        type_of_test = {}
        keys =  self.feature_1, self.feature_2
        for key in keys:
            type_of_test[key] = self._get_column_type(self.df.copy(), key)
        return type_of_test


    @staticmethod
    def _get_decision(condition):
        if condition:
            return "Reject H0,There is a relationship between 2 categorical variables"
        else:
            return "Retain H0,There is no relationship between 2 categorical variables"


    def _hypothsesis_chi_two(self, alpha = 0.05):
        df_contingency = pd.crosstab(self.df.copy()[self.feature_1], self.df.copy()[self.feature_1])
        observed_values = df_contingency.values 
        expected_values = stats.chi2_contingency(df_contingency)[3]
        no_of_rows, no_of_columns = df_contingency.shape
        degree_of_freedom = (no_of_rows - 1) * ( no_of_columns - 1)
        chi_square = sum([(o-e)**2./e for o,e in zip(observed_values, expected_values)])
        chi_square_statistic = chi_square[0] + chi_square[1]
        critical_value = chi2.ppf(q=1-alpha, df=degree_of_freedom)
        p_value = 1 - chi2.cdf(x=chi_square_statistic, df=degree_of_freedom)
        result = {
            'Significance level': alpha,
            'Degree of Freedom,degree_of_freedom': degree_of_freedom,
            'chi-square statistic': chi_square_statistic,
            'critical_value': critical_value,
            'p-value': p_value
        }
        self.info = {**self.info, **result}
        condition = chi_square_statistic >= critical_value
        return self._get_decision(condition)

    def _hypothsesis_rel(self):
        df_contigency = self.df.copy()[[self.feature_1, self.feature_2]]
        ttest, p_value = stats.ttest_rel(df_contigency[self.feature_2], df_contigency[self.feature_1])
        condition = p_value < 0.05
        return self._get_decision(condition)


    def _hypothsesis_oneway(self):
        df_contigency = self.df.copy()[[self.feature_1, self.feature_2]].fillna(0)
        grps = pd.unique(df_contigency.cheveux.values)
        d_data = {grp: df_contigency['salaire'][df_contigency.cheveux == grp] for grp in grps}
        F, p_val = stats.f_oneway(d_data['blond'], d_data['brun'], d_data['chatain'], d_data['roux'])
        condition = p_val < 0.05
        return self._get_decision(condition)

    def test(self):
        u, v = self._get_type_of_test()
        if u and v:
            return self._hypothsesis_chi_two()
        if not u and not v:
            return self._hypothsesis_rel()
        return self._hypothsesis_oneway()
    

    
  
    




        



