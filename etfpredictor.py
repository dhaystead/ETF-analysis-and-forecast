import numpy as np
import pandas as pd
import os
from statsmodels.tsa.stattools import adfuller, pacf, acf
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import skew
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

os.chdir("ETFs")



def get_data(dataset_name):
     df= pd.read_csv(dataset_name)
     df = df.set_index('Date')
     df = df.drop(['OpenInt', 'Open'], axis = 1)
     return df

def rolling_mean(dataset, window):
    rolling_mean = dataset.rolling(window=window).mean().dropna()
    return rolling_mean
def rolling_std(dataset, window):
    rolling_std = dataset.rolling(window=window).std().dropna()
    return rolling_std

def plot_rolling_stats(dataset, window):
    param_data = dataset.Close
    roll_mean = rolling_mean(param_data, window)
    roll_std = rolling_std(param_data, window)
    plt.figure(figsize=(12, 8))
    plt.plot(param_data, color='blue',label='Original')
    plt.plot(roll_mean, color='red', label='Rolling Mean')
    plt.plot(roll_std, color='black', label = 'Rolling Std')
    plt.title('Rolling Mean and Standard Deviation')
    plt.legend(loc='best')
    plt.show(block=False)


def adf_test(dataset):
    dickey_fuller = adfuller(dataset, autolag='AIC')
    df_results = pd.Series(dickey_fuller[:5], index=['p-value','test statistic' , '# of lags', '# of observations', 'critical values'])
    return df_results

def data_analysis(dataset,window):
    plot_rolling_stats(dataset, window)
    print(adf_test(dataset))

def check_stationary(dataset):
    adf_results = adf_test(dataset)
    if adf_results['p-value'] > 0.05:
        return False
    return True

def log_transform(dataset):
    """Log transform"""
    log_dataset = np.log(dataset)
    return log_dataset

def quadratic_transform(dataset):
    """Quadratic transform"""
    quad_dataset = np.square(dataset)
    return quad_dataset

def diff_terms(timeseries):
    num_diffs= 0
    while True:
        dftest = adfuller(timeseries)
        if dftest[0] <= dftest[4]['5%']:
             break
        else:
            timeseries= np.diff(timeseries)
            num_diffs += 1
    return num_diffs

def automated_p_q_values(timeseries):
    p=0
    q=0
    lag_acf = acf(timeseries, nlags=20)
    lag_pacf = pacf(timeseries, nlags=20, method='ols')
    y=1.96/np.sqrt(len(timeseries))
    if lag_acf[0] < y:
        for i in lag_acf:
            if i < y:
                q += 1
            break
    elif lag_acf[0] > y:
        for j in lag_acf:
            if j > y:
                q += 1
            break
    elif lag_pacf[0] < y:
        for k in lag_pacf:
            if k < y:
                p += 1
            break
    elif lag_pacf[0] > y:
        for l in lag_pacf:
            if l > y:
                p += 1
            break
    return p, q

def get_arima_parameters(timeseries):
    p, d, q = 0, 0, 0
    if not check_stationary(timeseries):
        if skew(timeseries) > .10:
            timeseries = log_transform(timeseries)
        elif skew(timeseries) < -.10:
            timeseries = quadratic_transform(timeseries)
        d = diff_terms(timeseries)
        p, q = automated_p_q_values(timeseries)
    return p, d, q

def build_arima_model(dataset):
    p, d, q = get_arima_parameters(dataset)
    model = ARIMA(dataset, (p, d, q))
    model_fit = model.fit(disp=0)
    return model_fit

def arima_prediction(dataset, window):
    timeseries = dataset.Close
    model_fit = build_arima_model(timeseries)
    forecast = model_fit.forecast(steps = window)
    return forecast


def svm_prediction(dataset):
    data = dataset.drop(['Close', 'High', 'Low'], axis =1)
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(data.values)
    df_x_norm = pd.DataFrame(x_norm, index=data.index, columns=data.columns)
    X_train, X_test, y_train, y_test = train_test_split(df_x_norm, data, test_size=.2, shuffle=False)
    kernels = {'linear', 'rbf', 'poly'}
    predict_lst = []
    mse_lst = []
    for i in kernels:
        if i == 'poly':
            degrees = {3, 4, 5}
            for j in degrees:
                model = SVR(kernel = i, degree = j)
                model_fit = model.fit(X_train, y_train.values.reshape(-1, 1))
                model_prediction = model_fit.predict(X_test)
                mse = mean_squared_error(y_test, model_prediction)
                mse_lst.append(mse)
                predict_lst.append([model_prediction, mse])
        else:
            model = SVR(kernel = i)
            model_fit = model.fit(X_train, y_train.values.reshape(-1, 1))
            model_prediction = model_fit.predict(X_test)
            mse = mean_squared_error(y_test, model_prediction)
            mse_lst.append(mse)
            predict_lst.append(model_prediction)
    min_mse = 0
    for i in range(1, len(mse_lst)):
        if mse_lst[min_mse] > mse_lst[i]:
            min_mse = i
    return predict_lst[min_mse], min_mse
