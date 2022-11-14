# -----------------------------------------------------------------------------
#
# Create logistic classifier for papers with or without novel microsporidia -
# host interactions
#
# Jason Jiang - Created: 2022/11/10
#               Last edited: 2022/11/14
#
# Mideo Lab - Microsporidia text mining
#
# -----------------------------------------------------------------------------

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

###############################################################################

# Load in labelled interactions dataset
with open('labelled_interactions.pickle', 'rb') as f:
    labelled_dataset = pickle.load(f)

data = np.array([entry[2].astype(int) for entry in labelled_dataset])
labels = np.array([int(entry[1]) for entry in labelled_dataset])

###############################################################################

## Rule-based baseline

def naive_classifier(x: np.ndarray) -> np.ndarray:
    # For a 2D array of candidate word appearances in text, return a 1D array
    # with 1 or 0 for each original row
    #
    # 1 = >=1 candidate word appears in text
    # 0 = no candidate words appear in text
    return np.sum(x, axis=1).astype(dtype=bool).astype(dtype=int)

###############################################################################

## Set up 10-fold cross-validation training/testing sets for our dataset

kf = KFold(n_splits=10, shuffle=True, random_state=42)
kf_split = tuple(kf.split(labelled_dataset))

###############################################################################

## Evaluate our rule-based baseline for classifying positive/negative documents

precision_pos = []
precision_neg = []
recall_pos = []
recall_neg = []
f1_pos = []
f1_neg = []

for train, test in kf_split:
    # only need test data, as we don't need to train our rule-based baseline
    test_data, test_labels = data[test], labels[test]
    naive_preds = naive_classifier(test_data)

    # create zipped list, where each tuple has predicted label and actual label
    zipped_pred_and_labels = list(zip(list(test_labels), list(naive_preds)))

    # positives:
    # tp: label = 1 and pred = 1
    # fp: label = 0 and pred = 1
    # fn: label = 1 and pred = 0
    tp_pos = np.sum([x[0] == 1 and x[1] == 1 for x in zipped_pred_and_labels])
    fp_pos = np.sum([x[0] == 0 and x[1] == 1 for x in zipped_pred_and_labels])
    fn_pos = np.sum([x[0] == 1 and x[1] == 0 for x in zipped_pred_and_labels])

    precision_pos_ = (tp_pos / (tp_pos + fp_pos)) * 100
    recall_pos_ = (tp_pos / (tp_pos + fn_pos)) * 100

    precision_pos.append(precision_pos_)
    recall_pos.append(recall_pos_)
    f1_pos.append(2 * ((precision_pos_ * recall_pos_) / (precision_pos_ + recall_pos_)))

    # negatives:
    # tp: label = 0 and pred = 0
    # fp: label = 1 and pred = 0
    # fn: label = 0 and pred = 1
    tp_neg = np.sum([x[0] == 0 and x[1] == 0 for x in zipped_pred_and_labels])
    fp_neg = np.sum([x[0] == 1 and x[1] == 0 for x in zipped_pred_and_labels])
    fn_neg = np.sum([x[0] == 0 and x[1] == 1 for x in zipped_pred_and_labels])

    precision_neg_ = (tp_neg / (tp_neg + fp_neg)) * 100
    recall_neg_ = (tp_neg / (tp_neg + fn_neg)) * 100

    precision_neg.append(precision_neg_)
    recall_neg.append(recall_neg_)
    f1_neg.append(2 * ((precision_neg_ * recall_neg_) / (precision_neg_ + recall_neg_)))

baseline_performance_df = dict(pd.DataFrame({
    'precision_pos': precision_pos,
    'precision_neg': precision_neg,
    'recall_pos': recall_pos,
    'recall_neg': recall_neg,
    'f1_pos': [0.0 if np.isnan(f1) else f1 for f1 in f1_pos],
    'f1_neg': [0.0 if np.isnan(f1) else f1 for f1 in f1_neg]
}).mean())


###############################################################################

## Logistic regression models for predicting positive/negative texts

# Set up class weights for positive and negative classes for parameter grid
# search
param_grid = [{0: 8.0, 1: 1.0}, {0: 4.0, 1: 1.0}, {0: 2.0, 1: 1.0},
              {0: 1.0, 1: 1.0}, {0: 1.0, 1: 2.0}, {0: 1.0, 1: 4.0},
              {0: 1.0, 1: 8.0}]

# initialize columns for dataframe of model performance metrics
class_weights = []
n_pos = []
n_neg = []
precision_pos = []
precision_neg = []
recall_pos = []
recall_neg = []
f1_pos = []
f1_neg = []

for param in param_grid:
    i = 1

    # ensure we use the same 10-fold dataset split when testing each set of
    # class weights
    for train, test in kf_split:
        train_data, train_labels = data[train], labels[train]
        test_data, test_labels = data[test], labels[test]

        model = LogisticRegression(solver='liblinear', random_state=42,
                                   class_weight=param).fit(train_data,
                                                           train_labels)

        model_stats = classification_report(test_labels, model.predict(test_data),
                                            output_dict=True)

        class_weights.append(f"negative: {param[0]}, positive: {param[1]}")
        n_pos.append(model_stats['1']['support'])
        n_neg.append(model_stats['0']['support'])
        precision_pos.append(model_stats['1']['precision'])
        precision_neg.append(model_stats['0']['precision'])
        recall_pos.append(model_stats['1']['recall'])
        recall_neg.append(model_stats['0']['recall'])
        f1_pos.append(model_stats['1']['f1-score'])
        f1_neg.append(model_stats['0']['f1-score'])

        i += 1

###############################################################################

# Create dataframe of performance metrics for each logistic model
model_performance_df = pd.DataFrame({
    'class_weights': class_weights,
    'n_pos': n_pos,
    'n_neg': n_neg,
    'precision_pos': precision_pos,
    'precision_neg': precision_neg,
    'recall_pos': recall_pos,
    'recall_neg': recall_neg,
    'f1_pos': f1_pos,
    'f1_neg': f1_neg
})

# Dataframe for average model performance for each set of class weights
param_rankings = {'negative: 8.0, positive: 1.0': 0,
                  'negative: 4.0, positive: 1.0': 1,
                  'negative: 2.0, positive: 1.0': 2,
                  'negative: 1.0, positive: 1.0': 3,
                  'negative: 1.0, positive: 2.0': 4,
                  'negative: 1.0, positive: 4.0': 5,
                  'negative: 1.0, positive: 8.0': 6}

avg_model_performance = \
    model_performance_df.groupby(['class_weights']).mean().reset_index().sort_values(
        by='class_weights', key=lambda col: col.map(param_rankings)
    )