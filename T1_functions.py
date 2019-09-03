import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from tqdm import tqdm_notebook
from statsmodels.tsa.api import Holt


def create_city_df(city, df_counts):    
    df_city = pd.DataFrame(df_counts[city].copy())
    df_city.columns = ["y"]
    df_city = df_city.dropna()
    print(city + ': ' + str(len(df_city)) + ' data points')
    return df_city


def plot_ts_diff(y, lags, city, plot_color, log_binary):
    num_diff = 1
    figsize=(12, 7)
    fig = plt.figure(figsize=figsize)
    layout = (num_diff+1, 3)
    
    if log_binary == 1:
        log_title = 'Log'
    else:
        log_title =''
        
    title_name = log_title + 'Time Series (' + city + ')'
    row_no = 0

    for loop_no in range(num_diff + 1):
        ts_ax = plt.subplot2grid(layout, (row_no, 0))
        acf_ax = plt.subplot2grid(layout, (row_no, 1))
        pacf_ax = plt.subplot2grid(layout, (row_no, 2))
        y.plot(ax=ts_ax, color=plot_color)  
        
        if city == 'Austin':
            ts_ax.set_title(title_name, fontweight="bold")
        else:
            p_value = sm.tsa.stattools.adfuller(y)[1]
            ts_ax.set_title(title_name + '\n Dickey-Fuller: p={0:.5f}'.format(p_value), fontweight="bold")
        
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        plt.xlabel('lags')
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.xlabel('lags')
        plt.tight_layout()

        y = y - y.shift(1)
        y = y[1:]
        title_name = 'First Difference ' + title_name
        row_no = row_no + 1


def MAPE(y_actual, y_prediction): 
    return np.mean(np.abs((y_actual - y_prediction) / y_actual)) * 100


def WalkForwardCV_ARIMA(param, X):
    
    n_train = len(X)//2
    n_records = len(X)
    error_list = []
    aic_list = []

    for i in range(n_train, n_records):
        train, test = X[0:i], X[i:i+1]
        
        try:
            model = sm.tsa.statespace.SARIMAX(train, order=(param[0], param[2], param[1])
                                           , enforce_stationarity = False
                                           , enforce_invertibility = False).fit(disp=-1)
        except:
            continue
            
        # predict next day 
        predictions = model.predict(start = len(train), end = len(train) + 1)

        # calculate MAPE error 
        error = MAPE(predictions, test)
        error_list.append(error)
        
        # obtain AIC
        aic_list.append(model.aic)

    return np.mean(error_list), np.mean(aic_list), model




def optimizeSARIMA(data, parameters_list):
    
    results = []

    for param in tqdm_notebook(parameters_list):
        error, aic, model = WalkForwardCV_ARIMA(param, data)
        
        results.append([param, error, aic])


    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'MAPE' ,'AIC']
    
    # sorting in ascending order, the lower error is - the better
    result_table = result_table.sort_values(by='MAPE', ascending=True).reset_index(drop=True)
    
    return result_table



def prediction_plot(predictions, actual, city, color_plot):
    
    plt.figure()
    predictions.plot(color = color_plot, alpha = 0.3)
    actual.plot(color = color_plot)
    plt.legend(['Prediction','Actual'])
    plt.title(city + ' Order Volume', fontsize = 15);
    plt.xlabel('Date', fontsize = 13);
    plt.ylabel('Order Count', fontsize = 13);
    plt.grid(True)
    plt.scatter(predictions.index, predictions, color = color_plot, alpha = 0.3)
    plt.scatter(actual.index, actual, color = color_plot);
    
    
def WalkForwardCV_HOLT(param, X):
    
    n_train = len(X)//2
    n_records = len(X)
    error_list = []
    aic_list = []

    for i in range(n_train, n_records):

        # Split train and test
        train, test = X[0:i], X[i:i+1]

        # Fit Holt's linear model
        fit1 = Holt(train).fit(smoothing_level= param[0]
                                            , smoothing_slope = param[1])
        # predict next day
        fcast1 = fit1.forecast(1)

        # calculate error 
        error = MAPE(fcast1, test)
        error_list.append(error)

        # obtain AIC
        aic_list.append(fit1.aic)

    return np.mean(error_list), np.mean(aic_list), fit1



def optimizeHOLT(parameters_list, data):
    
    results = []

    for param in tqdm_notebook(parameters_list):
        error, aic, model = WalkForwardCV_HOLT(param, data)
        
        results.append([param, error, aic])
        

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'MAPE' ,'AIC']
    
    # sorting in ascending order, the lower error is - the better
    result_table = result_table.sort_values(by='MAPE', ascending=True).reset_index(drop=True)
    
    return result_table


def naive_persistence(data):

    dataframe = pd.concat([data.shift(1), data], axis=1)
    X = dataframe.values
    test_score = MAPE(X[1:,0], X[1:,1])
    
    return test_score
    
