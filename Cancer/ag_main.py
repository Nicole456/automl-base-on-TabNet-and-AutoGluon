#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/23 8:32
# @Author  : iszhang
# @Email   : 
# @File    : ag_main.py
# @software: PyCharm

from autogluon import TabularPrediction as task
import sys

sys.path.append('C:/Users/ThinkPad/PycharmProjects/TabNet&AutoGluon/utils')
import utils.data_utils as data_utils

# pd.set_option('display.max_columns', None)
train_data = task.Dataset(file_path='../Data/5.Haberman/haberman.csv')
label_column = 'status'
dir = 'agModels-predictClass'  # specifies folder where to store trained models

# print(train_data.head(10))
# print(train_data.info())
# print(train_data.describe())

if __name__ == '__main__':
    # predictor = task.fit(train_data=train_data, label=label_column, output_directory=dir, time_limits=100)
    # results = predictor.fit_summary()
    # print("AutoGluon infers problem type is: ", predictor.problem_type)
    # print("AutoGluon identified the following types of features:")
    # print(predictor.feature_metadata)
    # # predictor.leaderboard(train_data, silent=True)
    # # print(results)
    time_limits = 60  # for quick demonstration only, you should set this to longest time you are willing to wait (in seconds)
    metric = 'roc_auc'  # specify your evaluation metric here
    predictor = task.fit(train_data=train_data, label=label_column, time_limits=time_limits)
    results = predictor.fit_summary()

    # print("AutoGluon infers problem type is: ", predictor.problem_type)
    # print("AutoGluon identified the following types of features:")
    # print(predictor.feature_metadata)

    # results.to_csv('111.csv')
    # data_utils.data_to_excel(results)
    # predictor.leaderboard(train_data, silent=True)
    # print(results)
