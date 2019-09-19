### Tools for linear regression ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Simluate data- create an array that 
def simulate_data(size1):
    x1= np.random.exponential(scale= 9000, size= size1)
    x2= np.random.poisson(lam= 15, size= size1)
    eps= np.random.randn(size1)
    x0=  np.ones(size1)
    # x1_vector= np.array(x1)
    # x2_vector= np.array(x2)
    X = np.column_stack((x0, x1, x2))
    beta = np.random.normal(0, 2.5, size= X.shape[1])
    y = X.dot(beta) + eps
    results = {"y": y,
               "X": X,
               "beta": beta}
    print(results)


def compare_models():
    """
    Compares output from different implementations of OLS.
    INPUT
        X (ndarray) the independent variables in matrix form
        y (array) the response variables vector
    RETURNS
        results (pandas.DataFrame) of estimated beta coefficients
    """
    pass


def load_hospital_data():
    """
    Loads the hospital charges data set found at data.gov.
    INPUT
        path_to_data (str) indicates the filepath to the hospital charge data (csv)
    RETURNS
        clean_df (pandas.DataFrame) containing the cleaned and formatted dataset for regression
    """
    pass


def prepare_data():
    """
    Prepares hospital data for regression (basically turns df into X and y).
    INPUT
        df (pandas.DataFrame) the hospital dataset
    RETURNS
        data (dict) containing X design matrix and y response variable
    """
    pass


def run_hospital_regression():
    """
    Loads hospital charge data and runs OLS on it.
    INPUT
        path_to_data (str) filepath of the csv file
    RETURNS
        results (str) the statsmodels regression output
    """
    pass
 

### END ###