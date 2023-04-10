# -----------------------------------------------------------------------------
#
# Predict spore shapes from texts with rule-based and hybrid approaches
#
# Jason Jiang - Created: Mar/27/2023
#               Last edited: Apr/10/2023
#
# Mideo Lab - Microsporidia text mining
#
#
# -----------------------------------------------------------------------------

import spacy
from spacy.matcher import Matcher
import pandas as pd
import re
from typing import Set, Tuple

###############################################################################

## Global variables

nlp = spacy.load('en_core_web_sm')

SHAPES = {'ovoid', 'pyriform', 'cylindical', 'spherical', 'oval', 'rounded',
          'elongated', 'rod-shaped', 'bacilliform', 'egg-shaped', 'ovoidal'}

spore_matcher = Matcher(nlp.vocab)
spore_matcher.add('spore', [[{"LEMMA": {"REGEX": ".*spore.*"}}]])

###############################################################################

def main():
    shapes_df = pd.read_pickle('../../../data/spore_shapes_formatted.pkl')

    shapes_df['pred_shapes_rules'] = shapes_df.apply(
        lambda row: predict_shapes_rules(row['title_abstract']),
        axis=1
    )

    shapes_df['pred_shapes_hybrid'] = shapes_df.apply(
        lambda row: predict_shapes_hybrid(row['title_abstract']),
        axis=1
    )

    shapes_df[['tp_rules', 'fp_rules', 'fn_rules']] = shapes_df.apply(
        lambda row: get_tp_fp_fn(row['shapes_in_text'],
                                 row['pred_shapes_rules']),
        axis=1,
        result_type='expand'
    )

    shapes_df[['tp_hybrid', 'fp_hybrid', 'fn_hybrid']] = shapes_df.apply(
        lambda row: get_tp_fp_fn(row['shapes_in_text'],
                                 row['pred_shapes_hybrid']),
        axis=1,
        result_type='expand'
    )

    # Calculate performance metrics for rule-based approach
    # fp: incidental shape used to describe something else, like meront
    #     ex: row 11
    # fn: out-of-vocabulary shape descriptor
    #     ex: row 18
    precision_ = sum(shapes_df.tp_rules) / (sum(shapes_df.tp_rules) + sum(shapes_df.fp_rules)) 
    recall_ = sum(shapes_df.tp_rules) / (sum(shapes_df.tp_rules) + sum(shapes_df.fn_rules))
    f1_ = (2 * precision_ * recall_) / (precision_ + recall_)

    # 82.0% precision, 64.2% recall, 72.0% F1
    print(f"Precision for rules: {round(precision_ * 100, 1)}%")
    print(f"Recall for rules: {round(recall_ * 100, 1)}%")
    print(f"F1-score for rules: {round(f1_ * 100, 1)}%")

    # Calculate performance metrics for hybrid approach
    # fp: non-specific adjectives captured
    # fn: shape descriptors incorrectly tagged as nouns by pipeline
    #     ex: oval, ovoid frequently tagged as noun
    #         21% of all abstracts with recorded shapes have ovoid or oval
    precision = sum(shapes_df.tp_hybrid) / (sum(shapes_df.tp_hybrid) + sum(shapes_df.fp_hybrid)) 
    recall = sum(shapes_df.tp_hybrid) / (sum(shapes_df.tp_hybrid) + sum(shapes_df.fn_hybrid))
    f1 = (2 * precision * recall) / (precision + recall)

    # 83.7% precision, 83.3% recall, 83.5% F1
    print(f"Precision for hybrid: {round(precision * 100, 1)}%")
    print(f"Recall for hybrid: {round(recall * 100, 1)}%")
    print(f"F1-score for hybrid: {round(f1 * 100, 1)}%")

###############################################################################

## Helper functions

def predict_shapes_rules(txt: str) -> Set[str]:
    # Join together all sentences with spore mentions, as they may describe
    # spore shapes
    spore_sents = '. '.join([sent.lower() for sent in re.split('\. *', txt) \
                             if sent != '' and 'spore' in sent.lower()])

    shapes_present = set()
    for shape in SHAPES:
        if shape in spore_sents:
            shapes_present = shapes_present | {shape}

    return shapes_present


def predict_shapes_hybrid(txt: str) -> Set[str]:
    spore_sents = [sent for sent in nlp(txt).sents if spore_matcher(sent)]
    
    return set([tok.text.lower() for sent in spore_sents for tok in sent if \
                tok.pos_ == 'ADJ'])


def get_tp_fp_fn(recorded: Set[str], predicted: Set[str]) -> Tuple[int]:
    tp = 0
    seen_recorded = set()

    for pred in predicted:
        for rec in recorded:
            if pred in rec:
                tp += 1
                seen_recorded = seen_recorded | {rec}
                continue

    fp = len(predicted) - tp
    fn = len(recorded) - len(seen_recorded)

    return tp, fp, fn

###############################################################################

if __name__ == '__main__':
    pass