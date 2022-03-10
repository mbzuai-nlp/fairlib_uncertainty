import pandas as pd
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
import numpy as np

import json
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

from itertools import combinations
from tqdm import tqdm

from collections import defaultdict 

def confusion_matrix_based_scores(cnf):
    """
    Implementation from https://stackoverflow.com/a/43331484
    """
    FP = cnf.sum(axis=0) - np.diag(cnf) + 1e-5
    FN = cnf.sum(axis=1) - np.diag(cnf) + 1e-5
    TP = np.diag(cnf) + 1e-5
    TN = cnf.sum() - (FP + FN + TP) + 1e-5

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    # Positive Prediction Rates
    PPR = (TP+FP)/(TP+FP+FN+TN)

    return {
        "TPR":TPR,
        "TNR":TNR,
        "PPV":PPV,
        "NPV":NPV,
        "FPR":FPR,
        "FNR":FNR,
        "FDR":FDR,
        "ACC":ACC,
        "PPR":PPR,
    }

def power_mean(series, p):
    if p>50:
        return max(series)
    elif p<50:
        return min(series)
    else:
        total = np.mean(np.power(series, p))
        return np.power(total, 1 / p)


def RMS_GAP(distinct_groups, all_scores, metric="TPR"):
    group_scores = []
    for gid in distinct_groups:
        # Save the TPR direct to the list 
        group_scores.append(all_scores[gid][metric]) 

    Scores = np.stack(group_scores, axis = 1)
    # Calculate GAP
    score_gaps = Scores - all_scores["overall"][metric].reshape(-1,1)
    # Sum over gaps of all protected groups within each class
    score_gaps = np.sum(abs(score_gaps),axis=1)
    # RMS of each class
    score_gaps = np.sqrt(np.mean(score_gaps**2))

    return score_gaps

def gap_eval_scores(y_pred, y_true, protected_attribute, metrics=["TPR","FPR","PPR"]):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    protected_attribute = np.array(protected_attribute)

    all_scores = {}
    # Overall evaluation
    distinct_labels = [i for i in range(len(set(y_true)))]
    overall_confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=distinct_labels)
    all_scores["overall"] = confusion_matrix_based_scores(overall_confusion_matrix)

    # Group scores
    distinct_groups = [i for i in range(len(set(protected_attribute)))]
    for gid in distinct_groups:
        group_identifier = (protected_attribute ==gid)
        group_confusion_matrix = confusion_matrix(y_true=y_true[group_identifier], y_pred=y_pred[group_identifier], labels=distinct_labels)
        all_scores[gid] = confusion_matrix_based_scores(group_confusion_matrix)

    eval_scores = {
        "accuracy" : accuracy_score(y_true, y_pred),
        "macro_fscore" : f1_score(y_true, y_pred, average="macro"),
        "micro_fscore" : f1_score(y_true, y_pred, average="micro"),
    }

    for _metric in metrics:
        eval_scores["{}_GAP".format(_metric)] = RMS_GAP(distinct_groups=distinct_groups, all_scores=all_scores, metric=_metric)

    return metrics