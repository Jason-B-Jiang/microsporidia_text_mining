# -----------------------------------------------------------------------------
#
# Predict spore shapes from texts with rule-based and hybrid approaches
#
# Jason Jiang - Created: Mar/27/2023
#               Last edited: Mar/27/2023
#
# Mideo Lab - Microsporidia text mining
#
#
# -----------------------------------------------------------------------------

import spacy
from spacy.matcher import Matcher
import pandas as pd
import re
from typing import Set

###############################################################################

## Global variables

nlp = spacy.load('en_core_web_sm')

SHAPES = {'ovoid', 'pyriform', 'cylindical', 'spherical', 'oval', 'round',
          'elongate', 'rod', 'bacilliform', 'egg'}

spore_matcher = Matcher(nlp.vocab)
spore_matcher.add('spore', [[{"LEMMA": {"REGEX": ".*spore.*"}}]])

###############################################################################

def main():
    shapes_df = pd.read_csv('../../../data/spore_shapes_formatted.csv')

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
    
    return set([tok.lemma_ for sent in spore_sents for tok in sent if \
                tok.pos_ == 'ADJ'])

###############################################################################

if __name__ == '__main__':
    pass