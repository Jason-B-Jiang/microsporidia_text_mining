# -----------------------------------------------------------------------------
#
# Prepare and format polar coil data for text mining
#
# Jason Jiang - Created: Feb/02/2023
#               Last edited: Feb/07/2023
#
# Mideo Lab - Microsporidia text mining
#
#
# -----------------------------------------------------------------------------

import spacy
import re
import pandas as pd
import numpy as np
import os

###############################################################################

nlp = spacy.load('en_core_web_sm')  # initialize spacy's small language model

###############################################################################

def main():
    # load in cleaned microsporidia dataset, and remove papers w/out abstracts
    microsp_data = \
        pd.read_csv('../../../data/microsporidia_species_and_abstracts.csv')

    # remove rows w/out paper abstracts
    microsp_data = microsp_data[microsp_data['abstract'].notnull()]

    # filter to only rows/papers with recorded polar tube coil data, and
    # only select relevant columns
    polar_coil_data = microsp_data[
        microsp_data[['pt_coils_range', 'pt_coils_avg']].notnull().any(1)
        ][['species', 'first_paper_title', 'abstract', 'pt_coils_range', 'pt_coils_avg']]

    # filter out rows where polar tube coil data is not in abstract
    polar_coil_data = \
        polar_coil_data[
            polar_coil_data['abstract'].map(lambda text: check_if_coil_data_in_text(text))
        ]

    # in cases where polar tube coil range is missing, fill in with average coil
    # count
    polar_coil_data['pt_coils_range'] = \
        polar_coil_data['pt_coils_range'].combine_first(polar_coil_data['pt_coils_avg'])

    # finally, add columns for manually correcting recorded polar coil ranges/
    # so they're all consistently formatted
    polar_coil_data['pt_coils_range_corrected'] = np.nan

    # save to data folder, under subfolder for polar tube coil data
    os.mkdir('../../../data/polar_coil_data')
    polar_coil_data.to_csv('../../../data/polar_coil_data/formatted_polar_coils.csv',
                           index=False)
    
    # load in manually corrected data (I copied the file created above after
    # I filled in the 'pt_coils_range_corrected' column)
    polar_tube_data_corrected = \
    	pd.read_csv('../../../data/polar_coil_data/formatted_polar_coils_corrected.csv')

    # replace pt_coils_range with pt_coils_range_corrected, where
    # pt_coils_range_corrected is not NaN
    polar_tube_data_corrected['pt_coils'] = \
        polar_tube_data_corrected['pt_coils_range_corrected'].combine_first(
            polar_tube_data_corrected['pt_coils_range']
        )
    
    # select relevant columns for use, and filter out rows where pt coil not in text
    polar_tube_data_corrected = \
        polar_tube_data_corrected[
            ['species', 'first_paper_title', 'abstract', 'pt_coils']
            ][polar_tube_data_corrected['pt_coils_range_corrected'] != 'NOT IN TEXT']

    polar_tube_data_corrected.to_csv('../../../data/polar_coil_data/polar_coils.csv',
                                     index=False)

###############################################################################

## Helper functions

def check_if_coil_data_in_text(text: str) -> bool:
    doc = nlp(text)
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        # found polar tube + coil + numerical value in sentence, meaning there's
        # likely polar tube coil data within this text
        if re.search('polar +(tube|tubule|filament)', sent_text) and \
            re.search('(coil|twist|turn|spire)', sent_text) and \
                np.any([tok.pos_ == 'NUM' for tok in sent]):
                return True

    return False

###############################################################################

if __name__ == '__main__':
    main()
