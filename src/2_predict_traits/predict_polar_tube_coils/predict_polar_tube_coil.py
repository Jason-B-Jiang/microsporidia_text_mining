
# -----------------------------------------------------------------------------
#
# Predict microsporidia polar tube coil measures from texts
#
# Jason Jiang - Created: Feb/07/2023
#               Last edited: Mar/09/2023
#
# Mideo Lab - Microsporidia text mining
#
#
# -----------------------------------------------------------------------------

import spacy
from spacy.matcher import Matcher
import numerizer  # for converting numeric tokens to numeric strings

import re
import pandas as pd
import numpy as np

from typing import List, Tuple

###############################################################################

nlp = spacy.load('en_core_web_sm')  # initialize spacy's small language model

# regex patterns for pure rule-based extraction of polar tube coils
PT_REGEX = 'polar +(tube|tubule|filament)'
COIL_REGEX = \
    "([0-9]+( *- *| *to *)[0-9]+ +(coil|twist|spire|turn)s?|[0-9]+ +(coil|spire|twist|turn)s?)"

# spaCy matcher for hybrid rule + ML based extraction of polar tube coils
matcher = Matcher(nlp.vocab)

coil_pattern = [
    {'POS': 'NUM'},
    {'TEXT': {'REGEX': "^(?!.m$).*"}, 'OP': '{,2}'},
    {'POS': 'NUM', 'OP': '?'},
    {'TEXT': {'REGEX': "^(?!.m$).*"}, 'OP': '{,2}'},
    {'LEMMA': {'REGEX': '(coil|spire|twist|turn)'}}
]

matcher.add('polar_coil', [coil_pattern])

###############################################################################

def main():
    pt_df = pd.read_csv('../../../data/polar_coil_data/polar_coils.csv')

    pt_df['pred_pt_coils'] = [predict_polar_tube_measures(txt) for txt in pt_df.abstract]
    pt_df['pred_pt_coils_rules'] = [predict_polar_tube_measures_rules(txt) for txt in pt_df.abstract]
    pt_df['pt_coils_formatted'] = [convert_recorded_pt_to_arrays(pt) for pt in pt_df.pt_coils]

    # combine rows with information from the same papers, so each row has polar
    # tube coil information for species coming from the same paper
    agg_dict = {
    'species': lambda x: '; '.join(x),
    'pt_coils': lambda x: '; '.join(x),
    'pred_pt_coils': lambda x: [row for row in x],
    'pred_pt_coils_rules': lambda x: [row for row in x],
    'pt_coils_formatted': lambda x: [row for row in x]
    }

    pt_df_grouped = \
        pt_df.groupby(['first_paper_title', 'abstract']).agg(agg_dict).reset_index()

    # calculate true positives, false positives and flase negatives for pure
    # rule-based polar tube coil predictions for each paper
    pt_df_grouped[['tp_rules', 'fp_rules', 'fn_rules']] = \
        pt_df_grouped.apply(lambda row: get_tp_fp_fn(row['pt_coils_formatted'],
                                   row['pred_pt_coils_rules']),
                    axis=1,
                    result_type='expand')

    # calculate true positives, false positives and false negatives for hybrid
    # polar tube coil predictions for each paper
    pt_df_grouped[['tp', 'fp', 'fn']] = \
        pt_df_grouped.apply(lambda row: get_tp_fp_fn(row['pt_coils_formatted'],
                                   row['pred_pt_coils']),
                    axis=1,
                    result_type='expand')

    # Calculate performance metrics for rule-based approach
    precision_ = sum(pt_df_grouped.tp_rules) / (sum(pt_df_grouped.tp_rules) + sum(pt_df_grouped.fp_rules)) 
    recall_ = sum(pt_df_grouped.tp_rules) / (sum(pt_df_grouped.tp_rules) + sum(pt_df_grouped.fn_rules))
    f1_ = (2 * precision_ * recall_) / (precision_ + recall_)

    # 83.7% precision, 83.3% recall, 83.5% F1
    print(f"Precision for rules: {round(precision_ * 100, 1)}%")
    print(f"Recall for rules: {round(recall_ * 100, 1)}%")
    print(f"F1-score for rules: {round(f1_ * 100, 1)}%")

    # Calculate performance metrics for hybrid approach
    precision = sum(pt_df_grouped.tp) / (sum(pt_df_grouped.tp) + sum(pt_df_grouped.fp)) 
    recall = sum(pt_df_grouped.tp) / (sum(pt_df_grouped.tp) + sum(pt_df_grouped.fn))
    f1 = (2 * precision * recall) / (precision + recall)

    # 83.7% precision, 83.3% recall, 83.5% F1
    print(f"Precision for hybrid: {round(precision * 100, 1)}%")
    print(f"Recall for hybrid: {round(recall * 100, 1)}%")
    print(f"F1-score for hybrid: {round(f1 * 100, 1)}%")
    
    # serialize the resulting dataframe to a pickle file
    pt_df_grouped.to_pickle('../../../results/pt_coil_rules/pt_coil_preds.pkl')

###############################################################################

## Helper functions

# Functions for predicting polar tube coil measures with rules
def predict_polar_tube_measures_rules(text: str) -> List[np.ndarray]:
    """
    Pure rule-based approach for predicting polar tube coils from texts
    """
    sents = text.lower().split(". ")  # rule-based sentencization
    pt_preds = []

    for sent in sents:
        if re.search(PT_REGEX, sent) and re.search(COIL_REGEX, sent):
            pt_preds.append(
                sum(
                    [turn_coil_text_to_array(m[0]) for m in \
                        re.findall(COIL_REGEX, sent)]
                        )
                    )

    return pt_preds

def turn_coil_text_to_array(coil_match: str) -> np.ndarray:
    nums = []
    for tok in re.split('-| +', coil_match):
        try:
            nums.append(float(tok))
        except ValueError:
            pass

    return np.array(nums)

def predict_polar_tube_measures(text: str) -> List[np.ndarray]:
    """
    Hybrid (rules + ML) approach for predicting polar tube coils from texts
    """
    doc = nlp(text)

    # extract coil measures from polar tube sentences, treating the sum of
    # all coil measures from each sentence as the coils for a distinct spore
    # class, or microsporidia species
    polar_coils = []

    for sent in doc.sents:
        if re.search(PT_REGEX, sent.text.lower()) and \
            matcher(sent):
            
            coil = get_sentence_coil_measure(sent)
            
            if coil.shape[0] > 0:
                polar_coils.append(coil)

    # for each sentence, treat as one "coil measure"
    # accumulate list of such measures, with each measure represented by a
    # length 1 - 2 numpy array (being either an individual measure or range)
    return polar_coils

def get_sentence_coil_measure(sent: spacy.tokens.span.Span) -> np.ndarray:
    # corner case: uncoiled polar tube = 0 coils
    if re.search("(uncoiled|noncoiled|non-coiled)", sent.text.lower()):
        return np.array([0.0])

    # get polar coil pattern matches within sentence
    matches = matcher(sent)

    # resolve overlapping matches, removing the shorter of overlapping matches
    # and only keeping the longest one
    resolved_matches = resolve_overlapping_matches(matches)

    # return matches for polar coil measures as an array of floats
    # length 1 array if a single value, length 2 if a range of coils is given
    return format_coil_matches_as_numbers(resolved_matches, sent)

def resolve_overlapping_matches(matches: List[Tuple[int]]) -> List[Tuple[int]]:
    # overlapping coil matches will have same span end within a sentence
    # so, take the longest coil match (which comes after the shorter span),
    # and set the shorter overlapping match to None
    for i in range(1, len(matches)):
        if matches[i - 1][2] == matches[i][2]:
            matches[i - 1] = None

    first_resolved = [m for m in matches if m is not None]
    for i in range(1, len(first_resolved)):
        if first_resolved[i - 1][2] > first_resolved[i][1]:
            first_resolved[i] = (first_resolved[i][0], first_resolved[i - 1][1], first_resolved[i][2])
            first_resolved[i - 1] = None
    
    return [match for match in first_resolved if match is not None]

def format_coil_matches_as_numbers(resolved_matches: List[Tuple[int]],
                                   sent: spacy.tokens.span.Span) -> np.ndarray:
    coil_measures = []

    for match in resolved_matches:
        coil_measures.extend(
            convert_coil_measure_to_numeric(sent[match[1] : match[2]])
        )

    if len(coil_measures) < 1:
        return np.array([])

    return sum(coil_measures)

def convert_coil_measure_to_numeric(match: spacy.tokens.span.Span) -> List[np.ndarray]:    
    numeric_toks = [tok for tok in match if tok.pos_ == 'NUM']

    try:
        if len(numeric_toks) == 2:
            if '/' in numeric_toks[1].text:
                # improperly numerized fraction
                # ex: 7 1/2 coils -> 1/2 can't be converted to float after numerization without ValueError
                fraction = float(numeric_toks[1].text.split("/")[0]) / float(numeric_toks[1].text.split("/")[1])
                return [np.array([float(numeric_toks[0]._.numerized) + fraction])]

            return [np.sort(np.array([float(tok._.numerized) for tok in numeric_toks]))]
        
        elif len(numeric_toks) == 1:
            if re.search('\\d+.\\d+', numeric_toks[0].text):
                # some kind of weirdly delimited coil range detected as a single numeric token
                split_ = re.split('[^0-9\\.]', numeric_toks[0].text)

                if len(split_) == 2:
                    return [
                        np.sort(np.array([float(numerizer.numerize(split_[0])), float(numerizer.numerize(split_[1]))]))
                        ]
                
                return []
            
            return [np.array([float(numeric_toks[0]._.numerized)])]

        return []

    except (ValueError, TypeError) as e:
        return []


# Functions for cleaning recorded coils data and evaluating predictions with
# recorded coils data

def convert_recorded_pt_to_arrays(pt_coils: str) -> str:
    return [np.array(re.split(' *[^\.\d] *', re.sub(' ?\(.+', '', pt))).astype('float') for \
        pt in pt_coils.split('; ')]

def get_tp_fp_fn(pt_coils: List[List[np.ndarray]],
                 pt_coils_pred: List[List[np.ndarray]]) -> Tuple[int]:
    # first, unnest all the lists
    pt_coils = [arr for lst in pt_coils for arr in lst]
    pt_coils_pred = [arr for lst in pt_coils_pred for arr in lst]

    tp = 0
    for pred in pt_coils_pred:
        for recorded in pt_coils:
            tp += np.array_equal(pred, recorded)

    fp = len(pt_coils_pred) - tp
    fn = len(pt_coils) - tp

    return tp, fp, fn

###############################################################################

# if __name__ == '__main__':
#     main()