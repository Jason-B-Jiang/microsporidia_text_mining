# -----------------------------------------------------------------------------
#
# Create logistic classifier for papers with or without novel microsporidia -
# host interactions
#
# Jason Jiang - Created: 2022/11/10
#               Last edited: 2022/11/25
#
# Mideo Lab - Microsporidia text mining
#
# -----------------------------------------------------------------------------

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

###############################################################################

# Load in labelled interactions dataset
with open('labelled_interactions.pickle', 'rb') as f:
    labelled_dataset = pickle.load(f)

data = np.array([entry['vector'] for entry in labelled_dataset])
labels = np.array([int(entry['label']) for entry in labelled_dataset])

###############################################################################

## Set up 10-fold cross-validation training/testing sets for our dataset

kf = KFold(n_splits=10, shuffle=True, random_state=42)
kf_split = tuple(kf.split(labelled_dataset))

###############################################################################

## Weighted lasso logistic regression model for predicting interactions

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

i = 0
models = {}

for param in param_grid:
    # ensure we use the same 10-fold dataset split when testing each set of
    # class weights
    for train, test in kf_split:
        train_data, train_labels = data[train], labels[train]
        test_data, test_labels = data[test], labels[test]

        model = LogisticRegression(solver='liblinear',
                                   random_state=42,
                                   class_weight=param,
                                   penalty='l1').fit(train_data,
                                                     train_labels)

        model_stats = classification_report(test_labels,
                                            model.predict(test_data),
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

        models[i] = model
        i += 1

###############################################################################

# Create dataframe of performance metrics for each logistic model
model_performance_df = pd.DataFrame({
    'weights': class_weights * 2,
    'precision': precision_pos + precision_neg,
    'recall': recall_pos + recall_neg,
    'f1': f1_pos + f1_neg,
    'class': ['pos'] * 70 + ['neg'] * 70
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
    model_performance_df.groupby(['weights', 'class']).mean().reset_index().sort_values(
        by='weights', key=lambda col: col.map(param_rankings)
    ).reset_index()

###############################################################################

## Plot performance metrics for trained logistic model

p1 = sns.barplot(data=avg_model_performance, x='weights', y='f1', hue='class')
p1.set(xlabel=None)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.savefig("f1.png", bbox_inches='tight')
plt.clf()

p2 = sns.barplot(data=avg_model_performance, x='weights', y='precision', hue='class')
p2.set(xlabel=None)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.savefig("precision.png", bbox_inches='tight')
plt.clf()

p3 = sns.barplot(data=avg_model_performance, x='weights', y='recall', hue='class')
p3.set(xlabel=None)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.savefig("recall.png", bbox_inches='tight')
plt.clf()