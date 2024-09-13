import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model_dt = DecisionTreeRegressor()


def dt_getr2(x_train, x_test, y_train, y_test):
    model_dt.fit(x_train, y_train)
    y_pred = model_dt.predict(x_test)
    y_pred = y_pred.reshape(-1, 1)
    r2 = r2_score(y_test, y_pred)
    return r2


def dt_predict(x_future):
    pred = model_dt.predict(x_future)
    return pred
