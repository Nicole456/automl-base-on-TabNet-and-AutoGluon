#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/23 8:32
# @Author  : iszhang
# @Email   : 
# @File    : ag_main.py
# @software: PyCharm


from autogluon import TabularPrediction as task

train_data = task.Dataset(file_path='../Data/5.Haberman/haberman.csv')
label_column = 'status'
dir = 'agModels-predictClass'  # specifies folder where to store trained models

if __name__ == '__main__':
    predictor = task.fit(train_data=train_data, label=label_column, output_directory=dir, time_limits=100)
    predictor.fit_summary()