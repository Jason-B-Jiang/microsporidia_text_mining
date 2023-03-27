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
import pandas as pd
import numpy as np
import re
from typing import Set

###############################################################################

## Global variables

nlp = spacy.load('en_core_web_sm')

SHAPES = {'ovoid', 'pyriform', 'cylindical', 'spherical', 'oval', 'round',
          'elongated', 'rod-shaped', 'bacilliform', 'egg-shaped'}

###############################################################################

def main():
    shapes_df = pd.read_csv('../../../data/spore_shapes_formatted.csv')

###############################################################################

## Helper functions

def predict_shapes_rules(txt: str) -> Set[str]:
    pass


def predict_shapes_hybrid(txt: str) -> Set[str]:
    pass

###############################################################################

if __name__ == '__main__':
    pass