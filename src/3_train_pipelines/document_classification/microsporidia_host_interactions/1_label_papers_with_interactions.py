# -----------------------------------------------------------------------------
#
# Label paper abstracts with presence or absence of novel microsporidia-host
# interactions
#
# Jason Jiang - Created: 2022/11/10
#               Last edited: 2022/11/10
#
# Mideo Lab - Microsporidia text mining
#
# -----------------------------------------------------------------------------

import pandas as pd
import spacy
from typing import Union
from collections import Counter
import numpy as np
import pickle

###############################################################################

## Label paper abstracts with presence/absence of novel interactions

def convert_matches_to_nan(matches: Union[str, float]) -> Union[str, float]:
    if pd.isnull(matches):
        return matches
    
    # if matches are all NA, then we have no matches, so return NaN
    matches_split = list(set(matches.split(' || ')))
    if len(matches_split) == 1 and matches_split[0] == 'NA':
        return float('nan')

    return matches

def paper_has_recorded_interactions(microsp_in_text_matches: Union[str, float],
                                    hosts_in_text_matches: Union[str, float]) ->\
                                        bool:
    microsp_in_text_matches = convert_matches_to_nan(microsp_in_text_matches)
    hosts_in_text_matches = convert_matches_to_nan(hosts_in_text_matches)

    if not isinstance(microsp_in_text_matches, float) and \
        not isinstance(hosts_in_text_matches, float):
        return True

    return False

labelled_papers = \
    pd.read_csv('../../../../data/formatted_host_microsp_names.csv')[
        ['title_abstract', 'microsp_in_text_matches', 'hosts_in_text_matches']
    ]

labelled_papers['interaction'] = labelled_papers.apply(
    lambda row: paper_has_recorded_interactions(row['microsp_in_text_matches'],
                                                row['hosts_in_text_matches']),
    axis=1
)

###############################################################################

## Get top 50 words in papers with novel microsporidia-host interactions

nlp = spacy.load('en_core_web_sm')  # use small model, just need tokenization

# concatenate all papers with novel interactions, and process with spaCy
doc = nlp(labelled_papers[labelled_papers['interaction']]['title_abstract'].str.cat(sep = ' '))

# get all token lemmas where lemma not in stop words, and token is a verb
candidate_words = [tok.lemma_ for tok in doc if not \
    tok.lemma_ in nlp.Defaults.stop_words and \
    tok.pos_ == 'VERB']

candidate_words_counted = Counter(candidate_words)

candidate_words_sorted = sorted(list(set(candidate_words)),
                                reverse=True,
                                key=lambda word: candidate_words_counted[word])

# manually go over top 50 candidate words, and pick which ones to exclude
# excluded words would be unlikely to indicate interaction, and would be
# describing something else (ex: Microsporidia properties)
candidate_words_to_exclude = ['measure', 'form', 'base', 'produce', 'arrange',
    'include', 'surround', 'use', 'divide', 'consist', 'appear', 'discuss',
    'suggest', 'propose', 'place', 'compose', 'fix', 'shape', 'relate',
    'know', '×', 'study', 'compare', 'differ', 'stain', 'coil', 'elongate',
    'belong', 'pack', 'enclose', 'sequence']

# exclude above words from our candidate words, for our final set of "predictor"
# words for predicting novel microsporidia-host associations in texts
predictor_words = np.array(list((set(candidate_words_sorted[:50]) - set(candidate_words_to_exclude))))

###############################################################################

## Create labelled dataset for paper abstracts, presence/abscence of novel
## microsporidia/host interactions, and appearance of each predictor word
## in the abstracts

def get_predictor_array(txt: str, predictor_words: np.array) -> np.array:
    doc = nlp(txt)
    candidate_words = list(set([tok.lemma_ for tok in doc if not \
        tok.lemma_ in nlp.Defaults.stop_words and \
            tok.pos_ == 'VERB']))

    # return 1d array of booleans, T/F for each predictor word depending
    # on if it appears in txt
    return np.in1d(predictor_words, candidate_words)

labelled_dataset = \
    [(labelled_papers['title_abstract'][i], labelled_papers['interaction'][i],
        get_predictor_array(labelled_papers['title_abstract'][i], predictor_words)) \
            for i in range(len(labelled_papers))]

# write this labelled dataset as a pickle object
with open('labelled_interactions.pickle', 'wb') as handle:
    pickle.dump(labelled_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)