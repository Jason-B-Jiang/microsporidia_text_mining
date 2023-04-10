# -----------------------------------------------------------------------------
#
# Prepare recorded spore shapes for text mining
#
# Jason Jiang - Created: Mar/27/2023
#               Last edited: Apr/10/2023
#
# Mideo Lab - Microsporidia text mining
#
#
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os.path
from typing import Set

###############################################################################

def main():
    shapes_df = \
        pd.read_csv('../../../data/microsporidia_species_and_abstracts.csv')[
            ['species', 'title_abstract', 'spore_shape']
        ].dropna(subset=['title_abstract'])  # filter rows without abstracts

    # clean up recorded shapes by removing parenthesized info and splitting
    # by semicolon delimiters
    shapes_df['spore_shape_formatted'] = \
        shapes_df['spore_shape'].str.lower().replace(
            ';$', '', regex=True).replace('\n*', '', regex=True).replace(
            ' \(*[^;\(\)]+\)', '', regex=True).str.split('; *', regex=True)

    # merge rows with information from the same paper
    agg_dict = {
        'species': lambda x: '; '.join(x),
        'spore_shape': lambda x: ' || '.join([s for s in x if not pd.isnull(s)]),
        'spore_shape_formatted': lambda x: combine_lists(x)
    }

    shapes_df_grouped = \
        shapes_df.groupby(['title_abstract']).agg(agg_dict).reset_index()

    # get shapes not found directly in texts; will need to manually verify
    # presence in texts
    shapes_df_grouped['shapes_not_in_text'] = shapes_df_grouped.apply(
        lambda row: get_shapes_not_in_text(row['spore_shape_formatted'],
                                           row['title_abstract']),
        axis=1
    )

    shapes_df_grouped['shapes_not_in_text_corrected'] = \
        [np.nan] * len(shapes_df_grouped)

    # write temporary csv to manually fill shapes_not_in_text_corrected column
    if os.path.isfile('temp.csv'):
        corrected = pd.read_csv('temp.csv')
        to_correct = \
            shapes_df_grouped[shapes_df_grouped['title_abstract'].isin(corrected.title_abstract)].reset_index(drop=True)
        
        to_correct['shapes_not_in_text_corrected'] = corrected['shapes_not_in_text_corrected']

        shapes_df_grouped_corrected = pd.concat(
            [
            to_correct.reset_index(drop=True),
            shapes_df_grouped[shapes_df_grouped['shapes_not_in_text'].isnull()].reset_index(drop=True)
             ]
        ).reset_index(drop=True)

        shapes_df_grouped_corrected['shapes_in_text'] = shapes_df_grouped_corrected.apply(
            lambda row: get_shapes_in_text(row['spore_shape_formatted'],
                                           row['shapes_not_in_text'],
                                           row['shapes_not_in_text_corrected']),
            axis=1
        )

        shapes_df_grouped_corrected.to_pickle('../../../data/spore_shapes_formatted.pkl')

    else:
        shapes_df_grouped[shapes_df_grouped['shapes_not_in_text'].notnull()].to_csv('temp.csv',
                                                                                    index=False)
        raise FileNotFoundError

###############################################################################

## Helper functions

def combine_lists(lst):
    # Credit to ChatGPT for this one
    lst = [x for x in lst if isinstance(x, list)]
    if len(lst) == 0:
        return np.nan
    else:
        return sum(lst, [])


def get_shapes_not_in_text(shapes: Set[str], text: str) -> Set[str]:
    if type(shapes) == float:
        return np.nan

    shapes_not_in_text = \
        {shape for shape in shapes if shape.lower() not in text.lower()}

    if len(shapes_not_in_text) < 1:
        return np.nan

    return shapes_not_in_text


def get_shapes_in_text(shape_formatted: Set[str],
                       not_in_text: Set[str],
                       corrected: Set[str]) -> str:
    if type(corrected) == float:
        corrected = set()
    else:
        corrected = set(corrected.lower().split('; '))

    if type(not_in_text) == float:
        not_in_text = set()

    if type(shape_formatted) == float:
        shape_formatted = set()

    return (set(shape_formatted) - not_in_text) | corrected


###############################################################################

if __name__ == '__main__':
    main()