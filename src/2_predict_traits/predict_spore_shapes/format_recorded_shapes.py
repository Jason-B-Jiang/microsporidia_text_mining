# -----------------------------------------------------------------------------
#
# Prepare recorded spore shapes for text mining
#
# Jason Jiang - Created: Mar/27/2023
#               Last edited: Mar/27/2023
#
# Mideo Lab - Microsporidia text mining
#
#
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np

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

    shapes_df_grouped.to_csv('../../../data/spore_shapes_formatted.csv')

###############################################################################

## Helper functions

def combine_lists(lst):
    # Credit to ChatGPT for this one
    lst = [x for x in lst if isinstance(x, list)]
    if len(lst) == 0:
        return np.nan
    else:
        return sum(lst, [])
    
###############################################################################

if __name__ == '__main__':
    main()