import numpy as np
# import data_gen

from load_save import *
from Classifer import rnn


def obj_fun(soln):
    X_train = load('cur_X_train')
    X_test = load('cur_X_test')
    y_train = load('cur_y_train')
    y_test = load('cur_y_test')

    # Feature selection
    soln = np.round(soln[1:])
    X_train = X_train[:, np.where(soln == 1)[0]]
    X_test = X_test[:, np.where(soln == 1)[0]]
    pred, met= rnn(X_train,X_test,y_train, y_test)
    fit = 1 / met[0]

    return fit

