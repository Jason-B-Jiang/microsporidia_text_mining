# -----------------------------------------------------------------------------
#
# Check % of tokens that are OOV across spaCy models
#
# Jason Jiang - Created: 2022/11/14
#               Last edited: 2022/11/14
#
# Mideo Lab - Microsporidia text mining
#
# -----------------------------------------------------------------------------

import spacy
import pandas as pd
import numpy as np

texts = pd.read_csv('../../data/microsporidia_species_and_abstracts.csv')
texts = texts['title_abstract'].dropna().unique()

# join all paper abstracts together as a single string
texts_str = ' '.join(texts)

################################################################################

## Check % of out of vocabulary tokens from texts_str for specified spaCy models
MODELS = ['en_core_web_lg', 'en_core_sci_lg']
models_oov = {}

for model in MODELS:
    nlp = spacy.load(model)
    doc = nlp(texts_str)
    models_oov[model] = \
        np.sum([tok.is_oov for tok in doc]) / len(doc)
