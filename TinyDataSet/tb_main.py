#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/23 8:32
# @Author  : iszhang
# @Email   : 
# @File    : tb_main.py
# @software: PyCharm

import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder

sys.path.append('C:/Users/ThinkPad/PycharmProjects/TabNet&AutoGluon/utils')
np.random.seed(0)
import utils.data_utils as data_utils
import utils.tb_train as tb

pd.set_option('display.max_columns', None)

"""
read data & preprocess
"""
# columns = ['age', 'sex', 'chest-pain-type', 'resting-blood-pressure', 'per-serum-cholestoral',
#            'fasting-blood-sugar', 'resting-electrocardiographic', 'max-heart-rate', 'exercise', 'depression',
#            'slope', 'vessels-num', 'type', 'condition']
# target = 'condition'
target = 'Label'
id = ['Id']
train = pd.read_csv('../Data/1.Cretio/train.tiny.csv', index_col=0)
# train.to_csv('../Data/3.Heart/heart.csv', index=False)

# print(train.head(10))
# print(train.info())
nunique = train.nunique()
types = train.dtypes
for col in train.columns:
    print(col, train[col].nunique(), train[col].count())

# 删除缺失值大于90%的变量
train.dropna(thresh=len(train) * 0.25, axis=1, inplace=True)
print(train.head())

for col in train.columns:
    if types[col] == 'object':
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")  # 填补缺失值
        train[col] = l_enc.fit_transform(train[col].values)
    elif nunique[col] < 50:
        l_enc = LabelEncoder()
        train[col] = train[col].fillna(sys.maxsize)  # 填补缺失值
        train[col] = l_enc.fit_transform(train[col].values)
        # train[col].fillna(train[col].mode(), inplace=True)
    else:
        train[col].fillna(train[col].mean(), inplace=True)

# train.to_csv('../Data/1.Cretio/train.tiny2.csv', index=False)
train.reset_index(inplace=True)

train_indices, valid_indices, test_indices = data_utils.split(train)
features, cat_idxs, cat_dims = data_utils.get_cat_info(train, target, 50)

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
