
# -----------------------------------------------------------------------------
#
# Predict localities from microsporidia texts
#
# Jason Jiang - Created: Mar/20/2023
#               Last edited: Mar/22/2023
#
# Mideo Lab - Microsporidia text mining
#
#
# -----------------------------------------------------------------------------

import spacy
from flashgeotext.geotext import GeoText
import pandas as pd
import re
from typing import List

###############################################################################

## Global variables
nlp = spacy.load('en_core_web_sm')
geotext = GeoText()

###############################################################################

def main():
    pass

###############################################################################

## Helper functions

def unnest_subregions_from_regions(locs: str) -> List[str]:
    locs = [loc.strip() for loc in locs.split('; ')]
    unnested_locs = []

    for loc in locs:
        subregions = re.findall('(?<=\().+(?=\))', loc)
        if len(subregions) > 0:
            unnested_locs.extend(subregions[0].split(' | '))

        unnested_locs.append(re.sub(' *\(.+', '', loc))

    return unnested_locs


def extract_locs_from_geotext_match(geo_match: dict) -> List[str]:
    cities = [city for city in geo_match['cities']]
    countries = [country for country in geo_match['countries']]

    return cities + countries


def normalize_localities_with_geotext(locs: List[str]) -> List[str]:
    normalized_locs = []
    for loc in locs:
        geo_match = extract_locs_from_geotext_match(geotext.extract(input_text=loc))

        if geo_match:
            normalized_locs.extend(geo_match)
        else:
            normalized_locs.append(loc)

    return normalized_locs


def predict_localities_with_geotext(txt: str) -> List[str]:
    geo_matches = geotext.extract(input_text=txt)
    return set(extract_locs_from_geotext_match(geo_matches))


def predict_localities_with_spacy(txt: str) -> List[str]:
    doc = nlp(txt)
    loc_ents = [ent.text for ent in doc.ents if ent.label_ in ['LOC', 'GPE']]

    return set(normalize_localities_with_geotext(loc_ents))

###############################################################################

if __name__ == '__main__':
    main()