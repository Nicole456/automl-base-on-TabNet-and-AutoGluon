#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/23 9:51
# @Author  : iszhang
# @Email   : 
# @File    : split.py
# @software: PyCharm

from sklearn.preprocessing import LabelEncoder
import numpy as np
import xlwt

np.random.seed(0)

def split(train):
    if "Set" not in train.columns:
        train["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(train.shape[0],))
    train_indices = train[train.Set == "train"].index
    valid_indices = train[train.Set == "valid"].index
    test_indices = train[train.Set == "test"].index
    return train_indices, valid_indices, test_indices

def get_category_info(train, min_cate_size):
    nunique = train.nunique()
    types = train.dtypes
    categorical_columns = []  # 代表list列表数据类型，列表是一种可变序列
    categorical_dims = {}  # dict字典数据类型，字典是Python中唯一内建的映射类型
    for col in train.columns:
        if col == 'Set':
            continue
        if types[col] == 'object' or nunique[col] < min_cate_size:
            print(col, train[col].nunique())
            l_enc = LabelEncoder()
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
    return categorical_columns, categorical_dims

def get_cat_info(train, target, min_cat_size):
    categorical_columns, categorical_dims = get_category_info(train, min_cat_size)
    unused_feat = ['Set']
    features = [col for col in train.columns if col not in unused_feat + [target]]
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
    return features, cat_idxs, cat_dims

def data_to_excel(data):
    # import xlwt
    file = xlwt.Workbook(encoding='utf-8')
    # 指定file以utf-8的格式打开
    table = file.add_sheet('data')
    # 指定打开的文件名

    # data = {
    #     "1": ["张三", 150, 120, 100],
    #     "2": ["李四", 90, 99, 95],
    #     "3": ["王五", 60, 66, 68]
    # }
    # 字典数据

    ldata = []
    num = [a for a in data]
    # for循环指定取出key值存入num中
    num.sort()
    # 字典数据取出后无需，需要先排序

    for x in num:
        # for循环将data字典中的键和值分批的保存在ldata中
        t = [int(x)]
        for a in data[x]:
            t.append(a)
        ldata.append(t)

    for i, p in enumerate(ldata):
        # 将数据写入文件,i是enumerate()函数返回的序号数
        for j, q in enumerate(p):
            # print i,j,q
            table.write(i, j, q)
    file.save('data.xlsx')
