import sys, os
import pandas as pd
import numpy as np
import random 
from sklearn.datasets import load_svmlight_file

class read_data:
    def __init__(self):
        self.test_fold = []

        self.valid_fold = []
        self.train_fold = []
    
    def read_mq2008(self, path):
        for i in range(1,6):
            X, y, qid = load_svmlight_file('{}/Fold{}/train.txt'.format(path,i), query_id=True)
            X_test, y_test, qid_test = load_svmlight_file('{}/Fold{}/test.txt'.format(path,i), query_id=True)
            X_vali, y_vali, qid_vali = load_svmlight_file('{}/Fold{}/vali.txt'.format(path,i), query_id=True)
            X = X.toarray()
            X_test = X_test.toarray()
            X_vali = X_vali.toarray() 
            self.train_fold.append((X, y, qid))
            self.valid_fold.append((X_vali,y_vali,qid_vali))
            self.test_fold.append((X_test,y_test,qid_test))

    def read_mq2007(self, path):
         for i in range(1,6):
            X, y, qid = load_svmlight_file('{}/Fold{}/train.txt'.format(path,i), query_id=True)
            X_test, y_test, qid_test = load_svmlight_file('{}/Fold{}/test.txt'.format(path,i), query_id=True)
            X_vali, y_vali, qid_vali = load_svmlight_file('{}/Fold{}/vali.txt'.format(path,i), query_id=True)
            X = X.toarray()
            X_test = X_test.toarray()
            X_vali = X_vali.toarray() 
            self.train_fold.append((X, y, qid))
            self.valid_fold.append((X_vali,y_vali,qid_vali))
            self.test_fold.append((X_test,y_test,qid_test))

    def get_fold(self, n):

        return self.train_fold[n], self.test_fold[n], self.valid_fold[n]