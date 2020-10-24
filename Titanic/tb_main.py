#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/22 20:09
# @Author  : iszhang
# @Email   :
# @File    : step3.py
# @software: PyCharm

from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import sys

sys.path.append('C:/Users/ThinkPad/PycharmProjects/TabNet&AutoGluon/utils')
np.random.seed(0)
import utils.data_utils as data_utils
import utils.tb_train as tb

np.random.seed(0)
import os
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)

## 合并数据集，便于清洗
train = pd.read_csv('../Data/2.titanic/train.csv')
test = pd.read_csv('../Data/2.titanic/test.csv')

if "Set" not in train.columns:
    train["Set"] = np.random.choice(["train", "valid"], p=[.9, .1], size=(train.shape[0],))
train_indices = train[train.Set == "train"].index
valid_indices = train[train.Set == "valid"].index
test_indices = test.index

full = train.append(test, ignore_index=True)  # 要添加的index不出现重复的情况，可以通过设置ignore_index=True来避免

"""
数据预处理
"""
## 平均值填充
## 年龄
full['Age'] = full['Age'].fillna(full['Age'].mean())
## 船票价格
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())
## 登船港口仅有两个缺失值，填众数.mode()[0]
full['Embarked'] = full['Embarked'].fillna(full['Embarked'].mode()[0])
## 船舱号缺失太多，填充U（unknow）
full['Cabin'] = full['Cabin'].fillna('U')

"""
特征工程
"""
##通过正则表达式，提取出相应的称呼
full['Title'] = full['Name'].str.extract(r',(.*?)\.', expand=False)
full['Cabin'] = full['Cabin'].str[0]
full['familySize'] = full['SibSp'] + full['Parch'] + 1
full['familyType'] = full['familySize'].map(
    lambda s: 'Large_Family' if s > 5 else ('Small_Family' if 2 <= s <= 4 else 'Single'))
# print(full.head())

full = full.drop(['PassengerId', 'Name', 'Ticket', 'Parch', 'SibSp', 'familySize'], axis=1)
print(full.head())
target = 'Survived'
train_row = 891
train = full.loc[0:train_row - 1, :]
test = full.loc[train_row, :]

train.to_csv(path_or_buf='../Data/2.titanic/train2.csv')

nunique = train.nunique()
types = train.dtypes
categorical_columns = []  # 代表list列表数据类型，列表是一种可变序列
categorical_dims = {}  # dict字典数据类型，字典是Python中唯一内建的映射类型

for col in train.columns:
    if col == 'Set':
        continue
    if types[col] == 'object' or nunique[col] < 10:
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)

print(train.head())

unused_feat = ['Set']
features = [col for col in train.columns if col not in unused_feat + [target]]
cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

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
