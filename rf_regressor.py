import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model_rf = RandomForestRegressor()


def rf_getr2(x_train, x_test, y_train, y_test):
    model_rf.fit(x_train, y_train)
    y_pred = model_rf.predict(x_test)
    y_pred = y_pred.reshape(-1, 1)
    r2 = r2_score(y_test, y_pred)
    return r2


def rf_predict(x_future):
    pred = model_rf.predict(x_future)
    return pred
