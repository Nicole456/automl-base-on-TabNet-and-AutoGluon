#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/23 8:33
# @Author  : iszhang
# @Email   : 
# @File    : tb_main.py
# @software: PyCharm


import pandas as pd
import numpy as np
import sys

sys.path.append('C:/Users/ThinkPad/PycharmProjects/TabNet&AutoGluon/utils')
np.random.seed(0)
import utils.data_utils as data_utils
import utils.tb_train as tb

pd.set_option('display.max_columns', None)
"""
preprocessing
"""

columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps',
           'deg-malig', 'breast', 'breast-quad', 'irradiat']
train = pd.read_csv('../Data/4.Cancer/breast-cancer.data', names=columns)
target = 'Class'

train.drop(index=train[(train['breast-quad'] == "?") | (train['node-caps'] == "?")].index, inplace=True)
# train.index = train.reset_index(drop=True)
train = train.reset_index(drop=True)
train.to_csv('../Data/4.Cancer/breast-cancer.csv', index=False)

# print(train.head(260))

train_indices, valid_indices, test_indices = data_utils.split(train)
features, cat_idxs, cat_dims = data_utils.get_cat_info(train, target, 10)

"""
Network parameters
"""
clf = tb.get_classifier(cat_idxs, cat_dims)

"""
Training
"""
X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices]

tb.fit(clf, X_train, y_train, X_valid, y_valid)
tb.tb_plt(clf)
tb.pred(clf, X_test, y_test, X_valid, y_valid)
