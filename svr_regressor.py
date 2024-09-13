import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model_svr = SVR()


def svr_getr2(x_train, x_test, y_train, y_test):
    model_svr.fit(x_train, y_train)
    y_pred = model_svr.predict(x_test)
    y_pred = y_pred.reshape(-1, 1)
    r2 = r2_score(y_test, y_pred)
    return r2


def svr_predict(x_future):
    pred = model_svr.predict(x_future)
    return pred
