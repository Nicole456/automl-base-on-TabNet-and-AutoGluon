#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/23 9:32
# @Author  : iszhang
# @Email   : 
# @File    : tb_train.py
# @software: PyCharm

from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

np.random.seed(0)
import os
from matplotlib import pyplot as plt

max_epochs = 1000 if not os.getenv("CI", False) else 2

def get_classifier(cat_idxs, cat_dims):
    clf = TabNetClassifier(cat_idxs=cat_idxs,
                           cat_dims=cat_dims,
                           cat_emb_dim=1,
                           optimizer_fn=torch.optim.Adam,
                           optimizer_params=dict(lr=2e-2),
                           scheduler_params={"step_size": 50,  # how to use learning rate scheduler
                                             "gamma": 0.9},
                           scheduler_fn=torch.optim.lr_scheduler.StepLR,
                           mask_type='entmax'  # "sparsemax"
                           )
    return clf

def tb_plt(clf):
    # plot losses
    plt.plot(clf.history['loss'], color='r', label='loss')
    # plot auc
    plt.plot(clf.history['train_auc'], color='g', label='train_auc')
    plt.plot(clf.history['valid_auc'], color='b', label='valid-auc')
    # plot learning rates
    plt.plot(clf.history['lr'], color='y', label='lr')
    plt.legend(loc='upper right')
    plt.xlabel('steps')
    plt.show()

def fit(clf, X_train, y_train, X_valid, y_valid):
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=max_epochs, patience=20,
        batch_size=1024, virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False
    )

def pred(clf, X_test, y_test, X_valid, y_valid):
    preds = clf.predict_proba(X_test)
    test_auc = roc_auc_score(y_score=preds[:, 1], y_true=y_test)
    preds_valid = clf.predict_proba(X_valid)
    valid_auc = roc_auc_score(y_score=preds_valid[:, 1], y_true=y_valid)
    print(f"BEST VALID SCORE : {clf.best_cost}")
    print(f"FINAL TEST SCORE : {test_auc}")
    # check that best weights are used
    assert np.isclose(valid_auc, np.max(clf.history['valid_auc']), atol=1e-6)
