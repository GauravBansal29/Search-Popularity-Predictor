from flask import Flask, request, jsonify
from pytrends.request import TrendReq
from pytrends import dailydata
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import train_test_split
import dt_regressor as dt
import rf_regressor as rf
import en_regressor as en
import pr_regressor as pr
import svr_regressor as svr

app = Flask(__name__)


# Selects the page for which a function is to be defined. Right now there will only be one page in your website.
def fetchData(search):

    requests_args = {
        'headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        }
    }

    # Only need to run this once, the rest of requests will use the same session.
    pytrends = TrendReq(hl='en-US', requests_args=requests_args)
    keywords = [search]

    pytrends.build_payload(keywords, timeframe='all')
    data = pytrends.interest_over_time()
    x = pd.DataFrame(data)
    x1 = data.iloc[:, 0]
    x1.to_csv('rd.csv')
    m = pd.read_csv('rd.csv')
    print(m)
    return m


def organise_data(m, time):
    m.columns = ['date', 'value']
    m['Output'] = m[['value']].shift(-time)
    return m


def findmax(a, b, c, d, e):
    mx = max(a, b, c, d)
    if mx == a:
        return 1
    if mx == b:
        return 2
    if mx == c:
        return 3
    if mx == d:
        return 4
    if mx == e:
        return 5


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    search = data['search']
    time = int(data['months'])

    received_data = fetchData(search)
    model_data = organise_data(received_data, time)

    X = np.array(model_data['value'])[:-time]
    Y = np.array(model_data['Output'])[:-time]
    print(X)
    print(Y)
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2,  random_state=5)
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    x_future = model_data['value']
    x_future = x_future.tail(time)
    x_future = np.array(x_future)
    x_future = x_future.reshape(-1, 1)

    # apply regression models to evaluate r2 scores
    # Decision Tree Regressor
    r2_dt = dt.dt_getr2(x_train, x_test, y_train, y_test)
    # Random Forest Regressor
    r2_rf = rf.rf_getr2(x_train, x_test, y_train, y_test)
    # Elastic Net Regressor
    r2_en = en.en_getr2(x_train, x_test, y_train, y_test)
    # Polynomial Regressor
    r2_pr = pr.pr_getr2(x_train, x_test, y_train, y_test)
    # SVR Regressor
    r2_svr = svr.svr_getr2(x_train, x_test, y_train, y_test)

    model_sel = findmax(r2_dt, r2_rf, r2_en, r2_pr, r2_svr)

    if model_sel == 1:
        pred = dt.dt_predict(x_future)
        model_final = "Decision Tree Regression"
    elif model_sel == 2:
        pred = rf.rf_predict(x_future)
        model_final = "Random Forest Regression"
    elif model_sel == 3:
        pred = en.en_predict(x_future)
        model_final = "Elastic Net Regression "
    elif model_sel == 4:
        pred = pr.pr_predict(x_future)
        model_final = "Polynomial Regression"
    else:
        pred = svr.svr_predict(x_future)
        model_final = "Support Vector Regression"

    r2_final = max(r2_dt, r2_rf, r2_en, r2_pr, r2_svr)

    print(pred)
    print(r2_final)

    print(model_final)
    pred = pred.tolist()
    prev = model_data['value'].tolist()
    return [pred, prev, r2_final, model_final]


# The above function returns the HTML code to be displayed on the page
if __name__ == '__main__':

    app.run()
