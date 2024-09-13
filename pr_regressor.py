import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

degree = 2
poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())


def pr_getr2(x_train, x_test, y_train, y_test):
    poly.fit(x_train, y_train)
    y_pred = poly.predict(x_test)
    y_pred = y_pred.reshape(-1, 1)
    r2 = r2_score(y_test, y_pred)
    return r2


def pr_predict(x_future):
    pred = poly.predict(x_future)
    return pred
