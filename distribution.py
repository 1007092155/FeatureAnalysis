import pandas as pd
import numpy as np
from math import sqrt
from supervisedTrans import series_to_supervised
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from model import svr_predict


def fig_show(test, train):
    plt.figure(figsize=(7, 5))
    plt.subplot(111)  # 三行一列,第一个图
    plt.plot(test, label='test')
    plt.plot(train, label='train')
    plt.grid(True)
    plt.legend(loc=0)  # 图例位置自动
    plt.axis('tight')
    plt.ylabel('speed')
    plt.title('data plot')
    plt.show()


def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])


def euclid_dist(test, train):
    return sqrt(sum((test-train)**2))


def DWTSimilar(test, train, num):
    distance = []
    for data in train:
        distance.append(DTWDistance(test, data[:6]))
    # distance = np.array(distance)
    indexes = np.argsort(distance)
    train = train[indexes]
    # fig_show(test, train[0])
    return train[:num]


class Distribution:
    lags = 6
    attrs = ['aveSpeed']
    up = pd.read_csv('data/up/ave_up(replaceMissingData).csv', encoding='utf-8')
    down = pd.read_csv('data/down/ave_down(replaceMissingData).csv', encoding='utf-8')
    left = pd.read_csv('data/left/ave_left(replaceMissingData).csv', encoding='utf-8')
    right = pd.read_csv('data/right/ave_right(replaceMissingData).csv', encoding='utf-8')

    x_scaler = MinMaxScaler(feature_range=(0, 1)).fit(right[attrs].values[:8352, :])
    x_train = x_scaler.transform(right[attrs].values[:8352, :]).reshape(1, -1)[0]
    x_train = series_to_supervised(pd.DataFrame(x_train), 6, 1)
    del x_train['var1(t)']

    y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(up[attrs].values[6:8352, :])
    y_train = y_scaler.transform(up[attrs].values[6:8352, :]).reshape(-1, 1).ravel()
    x_train['var1(t)'] = y_train
    train = np.array(x_train)

    x_test = x_scaler.transform(right[attrs].values[8346:, :]).reshape(1, -1)[0]
    x_test = series_to_supervised(pd.DataFrame(x_test), 6, 1)
    del x_test['var1(t)']
    # print(x_test.shape)
    y_test = up[attrs].values[8352:, :].reshape(1, -1)[0]
    test = np.array(x_test)

    pred = []
    for test_data in test:
        train_data = DWTSimilar(test_data, train, 5)
        np.random.shuffle(train_data)
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        pred.append(svr_predict(x_train, y_train, test_data.reshape(1, -1), y_scaler))
    print(pred)


    # train = np.array(x_train)
    # np.random.shuffle(train)
    # x_train = train[:, :-1]
    # y_train = train[:, -1]

    # x_test = x_scaler.transform(right[attrs].values[8346:, :]).reshape(1, -1)[0]
    # x_test = series_to_supervised(pd.DataFrame(x_test), 6, 1)
    # del x_test['var1(t)']
    # # print(x_test.shape)
    # y_test = up[attrs].values[8352:, :].reshape(1, -1)[0]

    #相关系数
    # df_up = up[attrs].values.flatten()
    # df_down = down[attrs].values.flatten()
    # df_left = left[attrs].values.flatten()
    # df_right = right[attrs].values.flatten()
    #
    # df_up_delay = up[attrs].values[:288, :].flatten()
    # df_down_delay = down[attrs].values[:288, :].flatten()
    # df_left_delay = left[attrs].values[:288, :].flatten()
    # df_right_delay = right[attrs].values[:288, :].flatten()
    #
    # up_down = np.array([df_up, df_down])
    # df_up_down = pd.DataFrame(up_down.T, columns=['up', 'down'])
    # up_down_delay = np.array([df_up_delay, df_down_delay])
    # df_up_down_delay = pd.DataFrame(up_down_delay.T, columns=['up', 'down'])
    # print('down_cov', df_up_down.up.cov(df_up_down.down), df_up_down_delay.up.cov(df_up_down_delay.down))
    # print('down_corr', df_up_down.up.corr(df_up_down.down), df_up_down_delay.up.corr(df_up_down_delay.down))
    #
    # up_left = np.array([df_up, df_left])
    # df_up_left = pd.DataFrame(up_left.T, columns=['up', 'left'])
    # up_left_delay = np.array([df_up_delay, df_left_delay])
    # df_up_left_delay = pd.DataFrame(up_left_delay.T, columns=['up', 'left'])
    # print('left_cov', df_up_left.up.cov(df_up_left.left), df_up_left_delay.up.cov(df_up_left_delay.left))
    # print('left_corr', df_up_left.up.corr(df_up_left.left), df_up_left_delay.up.corr(df_up_left_delay.left))
    #
    # up_right = np.array([df_up, df_right])
    # df_up_right = pd.DataFrame(up_right.T, columns=['up', 'right'])
    # up_right_delay = np.array([df_up_delay, df_right_delay])
    # df_up_right_delay = pd.DataFrame(up_right_delay.T, columns=['up', 'right'])
    # print('right_cov', df_up_right.up.cov(df_up_right.right), df_up_right_delay.up.cov(df_up_right_delay.right))
    # print('right_corr', df_up_right.up.corr(df_up_right.right), df_up_right_delay.up.corr(df_up_right_delay.right))

    # 数据分布图
    # fig, axes = plt.subplots(nrows=4, ncols=1)
    # up[attrs].hist(bins=100, ax=axes[0])
    # # up[attrs] = np.log1p(up[attrs])
    # down[attrs].hist(bins=100, ax=axes[1])
    # left[attrs].hist(bins=100, ax=axes[2])
    # right[attrs].hist(bins=100, ax=axes[3])
    # plt.legend()
    # plt.grid(True)
    # plt.show()