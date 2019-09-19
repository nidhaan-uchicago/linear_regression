### Tools for linear regression ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def simulate_data():
    """
    Simulates data for testing linear_regression models.
    INPUT
        nobs (int) the number of observations in the dataset
    RETURNS
        data (dict) contains X, y, and beta vectors.
    """


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


def load_hospital_data(path_to_data):
    """
    Loads the hospital charges data set found at data.gov.
    INPUT
        path_to_data (str) indicates the filepath to the hospital charge data (csv)
    RETURNS
        clean_df (pandas.DataFrame) containing the cleaned and formatted dataset for regression
    """
    clean_df = pd.read_csv(path_to_data)

    return clean_df


def prepare_data(df):
    """
    Prepares hospital data for regression (basically turns df into X and y).
    INPUT
        df (pandas.DataFrame) the hospital dataset
    RETURNS
        data (dict) containing X design matrix and y response variable
    """
    x1 = df["Average Covered Charges"]
    x2 = df["Total Discharges"]
    y = df["Average Medicare Payments"]

    data = pd.DataFrame[]


def run_hospital_regression(path_to_data):
    """
    Loads hospital charge data and runs OLS on it.
    INPUT
        path_to_data (str) filepath of the csv file
    RETURNS
        results (str) the statsmodels regression output
    """
    hospital = pd.read_csv(path_to_data)
    hospital.head()

    # x1 = average covered charges, x2 = number of hospital discharges associated with a given time period, y = average medicare payments
    x1 = hospital["Average Covered Charges"]
    x2 = hospital["Total Discharges"]
    y = hospital["Average Medicare Payments"]

    results = sm.OLS(y, x1, x2)

    # return
    return results
 

### END ###