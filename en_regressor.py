import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model_en = ElasticNet(alpha=1.0, l1_ratio=0.5)


def en_getr2(x_train, x_test, y_train, y_test):
    model_en.fit(x_train, y_train)
    y_pred = model_en.predict(x_test)
    y_pred = y_pred.reshape(-1, 1)
    r2 = r2_score(y_test, y_pred)
    return r2


def en_predict(x_future):
    pred = model_en.predict(x_future)
    return pred
