# -----------------------------------------------------------------------------
#
# Create logistic classifier for papers with or without novel microsporidia -
# host interactions
#
# Jason Jiang - Created: 2022/11/10
#               Last edited: 2022/11/10
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

# Set up 5-fold 80:20 train-test split for our dataset, and train a logistic
# model for each split
kf = KFold(n_splits=5)
i = 1

# initialize columns for dataframe of model performance metrics
model_name = []
overall_score = []
n_pos = []
n_neg = []
precision_pos = []
precision_neg = []
recall_pos = []
recall_neg = []
f1_pos = []
f1_neg = []

for train, test in kf.split(labelled_dataset):
    train_data, train_labels = data[train], labels[train]
    test_data, test_labels = data[test], labels[test]

    model = LogisticRegression(solver='liblinear', random_state=42).fit(train_data,
                                                                        train_labels)

    model_stats = classification_report(test_labels, model.predict(test_data),
                                        output_dict=True)

    model_name.append(f"model_{i}")
    overall_score.append(model.score(test_data, test_labels))
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
    'model': model_name,
    'overall_score': overall_score,
    'n_pos': n_pos,
    'n_neg': n_neg,
    'precision_pos': precision_pos,
    'precision_neg': precision_neg,
    'recall_pos': recall_pos,
    'recall_neg': recall_neg,
    'f1_pos': f1_pos,
    'f1_neg': f1_neg
})

# TODO:
# 1) get seaborn plots ready
# 2) look into accounting for class imbalances
#    https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/
# 3) check vocab overlap between text corpus and our different models
# 4) if logistic regression works better after accounting for class imbalances,
#    look into random forest classifiers