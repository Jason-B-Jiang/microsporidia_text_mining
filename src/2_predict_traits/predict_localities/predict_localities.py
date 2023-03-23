
# -----------------------------------------------------------------------------
#
# Predict localities from microsporidia texts
#
# Jason Jiang - Created: Mar/20/2023
#               Last edited: Mar/23/2023
#
# Mideo Lab - Microsporidia text mining
#
#
# -----------------------------------------------------------------------------

import spacy
from flashgeotext.geotext import GeoText
import pandas as pd
import re
from taxonerd import TaxoNERD
from typing import List, Set, Tuple

###############################################################################

## Global variables
nlp = spacy.load('en_core_web_sm')
taxonerd = TaxoNERD(prefer_gpu=False)
nlp_taxonerd = taxonerd.load(model='en_core_eco_md')
geotext = GeoText()

###############################################################################

def main():
    locality_df = pd.read_csv('../../../data/microsporidia_species_and_abstracts.csv')
    locality_df = \
        locality_df[['species', 'title_abstract', 'locality']].dropna(subset=['title_abstract'])

    locality_df['normalized_locality'] = locality_df.apply(
        lambda row: normalize_localities_with_geotext(unnest_subregions_from_regions(row['locality'])),
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

    locality_df['pred_locality_ml_no_taxons'] = locality_df.apply(
        lambda row: filter_out_taxonomic_localities(row['pred_locality_ml'],
                                                    row['title_abstract']),
        axis=1
    )

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
    
    locs = [loc.strip() for loc in re.split(' *; *|, *', locs)]
    unnested_locs = []

    for loc in locs:
        subregions = re.findall('(?<=\().+(?=\))', loc)
        if len(subregions) > 0:
            unnested_locs.extend(subregions[0].split(' | '))

        unnested_locs.append(re.sub(' *\(.+\) *', '', loc))

    return unnested_locs


def extract_locs_from_geotext_match(geo_match: dict) -> List[str]:
    cities = [city.lower() for city in geo_match['cities']]
    countries = [country.lower() for country in geo_match['countries']]

    return cities + countries


def normalize_localities_with_geotext(locs: List[str]) -> Set[str]:
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


def get_taxonerd_entities(txt: str) -> Set[str]:
    return [ent.text.lower() for ent in nlp_taxonerd(txt).ents]


def filter_out_taxonomic_localities(locs: Set[str], txt: str) -> Set[str]:
    taxonerd_ents = get_taxonerd_entities(txt)
    return set([loc for loc in locs if sum([loc in taxon for taxon in taxonerd_ents]) < 1])


def get_predicted_locality_tp_fp_fn(recorded_locs: Set[str],
                                    pred_locs: Set[str]) -> Tuple[int]:
    tp = len(recorded_locs.intersection(pred_locs))
    fp = len(pred_locs) - tp
    fn = len(recorded_locs) - tp

    return tp, fp, fn

###############################################################################

if __name__ == '__main__':
    main()