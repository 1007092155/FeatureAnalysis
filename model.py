import numpy as np
from sklearn import svm


def svr_predict(x_train, y_train, x_test):
    svr = svm.SVR(kernel='rbf')
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    svr.fit(x_train, y_train)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
    predict = svr.predict(x_test)

    return predict[0]
