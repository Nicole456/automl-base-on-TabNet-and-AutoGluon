#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/21 19:09
# @Author  : iszhang
# @Email   :
# @File    : step1.py
# @software: PyCharm

from autogluon import TabularPrediction as task

train_data = task.Dataset(file_path='../Data/2.titanic/train2.csv')
label_column = 'Survived'
dir = 'agModels-predictClass'  # specifies folder where to store trained models

# subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
# train_data = train_data.sample(n=subsample_size, random_state=0)
# print(train_data.head())
# print("Summary of class variable: \n", train_data[label_column].describe())
if __name__ == '__main__':
    predictor = task.fit(train_data=train_data, label=label_column, output_directory=dir, time_limits=100)
    predictor.fit_summary()