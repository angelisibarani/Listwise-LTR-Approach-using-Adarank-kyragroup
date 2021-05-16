import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.utils import check_X_y
import sys
import math as math


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]


    gain = 2 ** r - 1
    discounts = np.log2(np.arange(len(r)) + 2)

    return np.sum(gain / discounts)
    
            


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max





class NDCGScorer:

    def __init__(self, K=5):
        self.K = K

    def __call__(self, y_true, y_predict):
        sorted_y_index = np.argsort(y_true)
        y = np.take(y_predict, sorted_y_index)
        return ndcg_at_k(y,self.K)

class NDCGScorer_qid:
    def __init__(self, K=5):
        self.scorer = NDCGScorer(K)
        self.K = K
    def __call__(self, y_true, y_predict, qid):
        return self.get_scores(y_true, y_predict, qid)

    def get_scores(self, true_list, pred_l, qid):
        scores = []
   
        prv_qid = qid[0]
        mark_index = [0]
        for itr, a in enumerate(qid):
            if a != prv_qid:
                mark_index.append(itr)
                prv_qid = a
        mark_index.append(qid.shape[0])
        for start, end in zip(mark_index, mark_index[1:]):
    
            scores.append(self.scorer(true_list[start:end], pred_l[start:end]))

        return np.array(scores)
    
    
def get_ap(y_true, y_pred):
    order = np.argsort(y_pred)
    y_true = np.take(y_true, order)
    pos = 1 + np.where(y_true > 0)[0]
    n_rels = 1 + np.arange(len(pos))
    return np.mean(n_rels / pos) if len(pos) > 0 else 0

class map_scorer():
    
    def __call__(self, y_true, y_predict, qid):
        return self.get_scores(y_true, y_predict, qid)
    
    def get_scores(self, true_list, pred_l,qid):
        scores = []
    
        prv_qid = qid[0]
        mark_index = [0]
        for itr, a in enumerate(qid):
            if a != prv_qid:
                mark_index.append(itr)
                prv_qid = a
        mark_index.append(qid.shape[0])
        for start, end in zip(mark_index, mark_index[1:]):
      
            scores.append(get_ap(true_list[start:end], pred_l[start:end]))
        return np.array(scores)

