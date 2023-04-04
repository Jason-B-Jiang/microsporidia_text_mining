
# -----------------------------------------------------------------------------
#
# Predict localities from microsporidia texts
#
# Jason Jiang - Created: Mar/20/2023
#               Last edited: Apr/04/2023
#
# Mideo Lab - Microsporidia text mining
#
#
# -----------------------------------------------------------------------------

import spacy
from flashgeotext.geotext import GeoText
import pandas as pd
import numpy as np
import re
import os
# from taxonerd import TaxoNERD
from typing import List, Set, Tuple

###############################################################################

## Global variables
nlp = spacy.load('en_core_web_sm')
# taxonerd = TaxoNERD(prefer_gpu=False)
# nlp_taxonerd = taxonerd.load(model='en_core_eco_md')
geotext = GeoText()

###############################################################################

def main():
    locality_df = pd.read_csv('../../../data/microsporidia_species_and_abstracts.csv')
    locality_df = \
        locality_df[['species', 'title_abstract', 'locality']].dropna(subset=['title_abstract'])

    locality_df['unnested_locality'] = locality_df.apply(
        lambda row: unnest_subregions_from_regions(row['locality']),
        axis=1
    )

    locality_df['locality_not_in_text'] = locality_df.apply(
        lambda row: get_locs_not_in_text(row['unnested_locality'], row['title_abstract']),
        axis=1
    )

    locality_df = locality_df.assign(
        locality_not_in_text_corrected = [np.nan] * len(locality_df)
        )
    
    # locality_df.to_csv('temp.csv')
    corrected = pd.read_csv('temp.csv')
    if all(pd.isnull(corrected['locality_not_in_text_corrected'])):
        raise FileNotFoundError

    locality_df['locality_not_in_text_corrected'] = \
        corrected['locality_not_in_text_corrected']

    locality_df['locality_corrected'] = locality_df.apply(
        lambda row: get_corrected_locality(row['unnested_locality'],
                                           row['locality_not_in_text'],
                                           row['locality_not_in_text_corrected']),
        axis=1
    )

    # aggregate localities from the same papers
    agg_dict = {
        'species': lambda x: '; '.join(x),
        'locality': lambda x: ' || '.join([l for l in x if type(l) != float]),
        'unnested_locality': lambda x: set.union(*[set(row) if type(row) != float else set() for row in x]),
        'locality_not_in_text': lambda x: set.union(*[row if type(row) != float else set() for row in x]),
        'locality_not_in_text_corrected': lambda x: ' || '.join([l for l in x if type(l) != float]),
        'locality_corrected': lambda x: set.union(*[row if type(row) != float else set() for row in x])
        }

    locality_df = \
        locality_df.groupby(['title_abstract']).agg(agg_dict).reset_index()

    locality_df['normalized_locality'] = locality_df.apply(
        lambda row: normalize_localities_with_geotext(row['locality_corrected']),
        axis=1
    )

    locality_df['pred_locality_rules'] = locality_df.apply(
        lambda row: predict_localities_with_geotext(row['title_abstract']),
        axis=1
    )

    locality_df['pred_locality_ml'] = locality_df.apply(
        lambda row: predict_localities_with_spacy(row['title_abstract']),
        axis=1
    )

    # locality_df['pred_locality_ml_no_taxons'] = locality_df.apply(
    #     lambda row: filter_out_taxonomic_localities(row['pred_locality_ml'],
    #                                                 row['title_abstract']),
    #     axis=1
    # )

    locality_df[['tp_rules', 'fp_rules', 'fn_rules']] = locality_df.apply(
        lambda row: get_predicted_locality_tp_fp_fn(row['normalized_locality'],
                                                    row['pred_locality_rules']),
        axis=1,
        result_type='expand'
    )

    locality_df[['tp_ml', 'fp_ml', 'fn_ml']] = locality_df.apply(
        lambda row: get_predicted_locality_tp_fp_fn(row['normalized_locality'],
                                                    row['pred_locality_ml']),
        axis=1,
        result_type='expand'
    )

    # calculate precision, recall and f1 for rules and hybrid approach
    precision_rules = sum(locality_df.tp_rules) / (sum(locality_df.tp_rules) + sum(locality_df.fp_rules))
    recall_rules = sum(locality_df.tp_rules) / (sum(locality_df.tp_rules) + sum(locality_df.fn_rules))
    f1_rules = (2 * precision_rules * recall_rules) / (precision_rules + recall_rules)

    # 46.4% precision, 28.2% recall, 35.1% F1
    print('\n#------------------------------------------------------------#\n')
    print(f"Precision for rules: {round(precision_rules * 100, 1)}%")
    print(f"Recall for rules: {round(recall_rules * 100, 1)}%")
    print(f"F1-score for rules: {round(f1_rules * 100, 1)}%")
    print('\n#------------------------------------------------------------#\n')

    precision_hybrid = sum(locality_df.tp_ml) / (sum(locality_df.tp_ml) + sum(locality_df.fp_ml))
    recall_hybrid = sum(locality_df.tp_ml) / (sum(locality_df.tp_ml) + sum(locality_df.fn_ml))
    f1_hybrid = (2 * precision_hybrid * recall_hybrid) / (precision_hybrid + recall_hybrid)

    # 16.0% precision, 31.5% recall, 21.2% F1
    print(f"Precision for hybrid: {round(precision_hybrid * 100, 1)}%")
    print(f"Recall for hybrid: {round(recall_hybrid * 100, 1)}%")
    print(f"F1-score for hybrid: {round(f1_hybrid * 100, 1)}%")
    print('\n#------------------------------------------------------------#\n')

###############################################################################

## Helper functions

def unnest_subregions_from_regions(locs: str) -> List[str]:
    if type(locs) == float:  # NaN for recorded localities
        return []
    
    locs = [loc.strip() for loc in re.split(' *; *', locs)]
    unnested_locs = []

    for loc in locs:
        subregions = re.findall('(?<=\().+(?=\))', loc)
        if len(subregions) > 0:
            unnested_locs.extend(
                [l for l in re.split('(, *| \| )', subregions[0]) \
                 if ',' not in l and '|' not in l and l != '?']
            )

        unnested_locs.append(re.sub(' *\(.+\) *', '', loc))

    return unnested_locs


def get_locs_not_in_text(locs: List[str], text: str) -> Set[str]:
    not_in_text = []
    for loc in locs:
        if loc not in text:
            not_in_text.append(loc)
    
    return set(not_in_text)


def extract_locs_from_geotext_match(geo_match: dict) -> List[str]:
    cities = [city.lower() for city in geo_match['cities']]
    countries = [country.lower() for country in geo_match['countries']]

    return cities + countries


def get_corrected_locality(locality: List[str], not_in_text: Set[str],
                           corrected: str) -> Set[str]:
    only_in_text = set(locality) - not_in_text
    
    # NaN for corrected
    if type(corrected) == float:
        return only_in_text
    
    corrected = set(corrected.split('; '))
    return only_in_text | corrected


def normalize_localities_with_geotext(locs: Set[str]) -> Set[str]:
    normalized_locs = []
    for loc in locs:
        geo_match = extract_locs_from_geotext_match(geotext.extract(input_text=loc))

        if geo_match:
            normalized_locs.extend(geo_match)
        else:
            if loc != '?':  # sometimes ? was recorded to indicate uncertainty, not subregion
                normalized_locs.append(loc.lower())

    return set(normalized_locs)


def predict_localities_with_geotext(txt: str) -> List[str]:
    geo_matches = geotext.extract(input_text=txt)
    return set(extract_locs_from_geotext_match(geo_matches))


def predict_localities_with_spacy(txt: str) -> Set[str]:
    doc = nlp(txt)
    loc_ents = [re.sub('[Tt]he ', '', ent.text) for ent in doc.ents if ent.label_ in ['LOC', 'GPE']]

    return normalize_localities_with_geotext(loc_ents)


# def get_taxonerd_entities(txt: str) -> Set[str]:
#     return [ent.text.lower() for ent in nlp_taxonerd(txt).ents]


# def filter_out_taxonomic_localities(locs: Set[str], txt: str) -> Set[str]:
#     taxonerd_ents = get_taxonerd_entities(txt)
#     return set([loc for loc in locs if sum([loc in taxon for taxon in taxonerd_ents]) < 1])


def get_predicted_locality_tp_fp_fn(recorded_locs: Set[str],
                                    pred_locs: Set[str]) -> Tuple[int]:
    tp = len(recorded_locs.intersection(pred_locs))
    fp = len(pred_locs) - tp
    fn = len(recorded_locs) - tp

    return tp, fp, fn

###############################################################################

if __name__ == '__main__':
    main()