import pandas as pd
import numpy as np
import math
from supervisedTrans import series_to_supervised
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm


def plot_results(y_true, y_preds):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2016-11-30 00:00'
    x = pd.date_range(d, periods=288, freq='5min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    ax.plot(x, y_preds, label='SVR')

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)

    print(y_pred)
    print(y_true)


class FindSimilar:
    attrs = ['aveSpeed']
    up = pd.read_csv('data/up/ave_up(replaceMissingData).csv', encoding='utf-8')
    down = pd.read_csv('data/down/ave_down(replaceMissingData).csv', encoding='utf-8')
    left = pd.read_csv('data/left/ave_left(replaceMissingData).csv', encoding='utf-8')
    right = pd.read_csv('data/right/ave_right(replaceMissingData).csv', encoding='utf-8')

    x_scaler = MinMaxScaler(feature_range=(0, 1)).fit(right[attrs].values[:8352, :])
    x_train = x_scaler.transform(right[attrs].values[:8352, :]).reshape(1, -1)[0]
    x_train = series_to_supervised(pd.DataFrame(x_train), 6, 1)
    del x_train['var1(t)']
    # print(x_train)
    y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(up[attrs].values[6:8352, :])
    y_train = y_scaler.transform(up[attrs].values[6:8352, :]).reshape(-1, 1).ravel()
    x_train['var1(t)'] = y_train
    train = np.array(x_train)
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_test = x_scaler.transform(right[attrs].values[8346:, :]).reshape(1, -1)[0]
    x_test = series_to_supervised(pd.DataFrame(x_test), 6, 1)
    del x_test['var1(t)']
    # print(x_test.shape)
    y_test = up[attrs].values[8352:, :].reshape(1, -1)[0]

    svr = svm.SVR(kernel='rbf')
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    svr.fit(x_train, y_train)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
    predicted = svr.predict(x_test)
    predicted = y_scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
    eva_regress(y_test, predicted)
    plot_results(y_test, predicted)


    # df_up = up[attrs].values[100:150, :]
    # df_down = down[attrs].values[93:143, :]
    # df_left = left[attrs].values[100:150, :]
    # df_right = right[attrs].values[100:150, :]

    # plt.figure(figsize=(7, 5))
    # plt.subplot(311)  # 三行一列,第一个图
    # plt.plot(df_up, label='up')
    # plt.grid(True)
    # plt.legend(loc=0)  # 图例位置自动
    # plt.axis('tight')
    # plt.ylabel('speed')
    # plt.title('up data plot')
    #
    # plt.subplot(312)  # 三行一列.第二个图
    # plt.plot(df_down, 'g', label='down')
    # plt.grid(True)
    # plt.legend(loc=0)
    # plt.xlabel('index')
    # plt.ylabel('speed')
    # plt.axis('tight')
    #
    # plt.subplot(313)  # 三行一列.第三个图
    # plt.plot(df_left, 'g', label='left')
    # plt.grid(True)
    # plt.legend(loc=0)
    # plt.xlabel('index')
    # plt.ylabel('speed')
    # plt.axis('tight')
    # plt.show()
