# -----------------------------------------------------------------------------
#
# Predict microsporidia localities from paper titles + abstracts: V2
#
# Jason Jiang - Created: 2022/05/25
#               Last edited: 2022/07/26
#
# Mideo Lab - Microsporidia text mining
#
#
# -----------------------------------------------------------------------------

from flashgeotext.geotext import GeoText
import spacy
import geocoder
from typing import Dict, List, Tuple, Optional
import pandas as pd
import re
import copy
from pathlib import Path
from taxonerd import TaxoNERD
from Levenshtein import distance as levenshtein_distance

################################################################################

## Initialize language models
nlp = spacy.load('en_core_web_md')
geotext = GeoText()

# use taxonerd to predict taxonomic entities in texts, and prevent these from
# being part of locality predictions (as spaCy tends to tag taxonomic names
# as geographical entities)
taxonerd = TaxoNERD(model="en_ner_eco_biobert", prefer_gpu=False, with_abbrev=True)

################################################################################

## Global variables
SPACY_LOCALITIES = ['FAC', 'GPE', 'LOC']  # spacy entities corresponding to localities

GEONAMES_CACHE = {}

USERNAME = 'jiangjas'  # fill in your own geonames username here

################################################################################

def main() -> None:
    # Load in formatted dataframe of microsporidia species data, from
    # src/1_format_data/3_misc_cleanup.R
    microsp_data = pd.read_csv('../../data/manually_format_multi_species_papers.csv')

    # Exclude species with >1 papers describing them (for now)
    microsp_data = microsp_data[microsp_data['num_papers'] < 2]

    # Make locality predictions using title_abstract column
    microsp_data = microsp_data.assign(
        # pred = predicted
        pred_locality = lambda df: df['title_abstract'].map(
            lambda txt: get_localities_string(predict_localities(txt))
        )
    )

    microsp_data = microsp_data.assign(
        locality_normalized = lambda df: df['locality'].map(
            lambda locs: get_localities_string(
                normalize_recorded_localities(locs)
                ), na_action='ignore'
        )
    )

    microsp_data[['species', 'title_abstract', 'locality', 'locality_normalized',
    'pred_locality']].to_csv(
        Path('../../results/microsp_locality_predictions.csv')
    )

################################################################################

## Helper functions for predicting regions + subregions from texts

def get_cached_geonames_results(loc: str):
    """Retrieve cached geonames search results for some putative location, loc.
    If loc isn't in cache, then get the geonames search results and store it in
    the cache for later.

    Sort geonames results by string similarity to loc (measured by Levenshtein
    distance), from highest to lowest similarity (lowest to highest string
    distance).

    Input:
        loc: string for some predicted location

    Output:
        instance of geonames search results class
    """
    if not loc in GEONAMES_CACHE:
        # this particular location not looked up in geonames yet, get top 50
        # search results (first page of results) and store in cache
        geonames_result = geocoder.geonames(loc, key=USERNAME, maxRows=50)
        GEONAMES_CACHE[loc] = geonames_result
    else:
        # fetch cached results
        geonames_result = GEONAMES_CACHE[loc]

    return sorted(geonames_result,
        key=lambda res: levenshtein_distance(loc, res.address))


def remove_leading_determinant(span: spacy.tokens.span.Span) -> str:
    """Remove leading determinant from a spacy span for a location entity, and
    return the resulting text.
    Ex: the Weddell Sea -> Weddell Sea

    Input:
        span: spaCy span representing a location entity
    
    Return:
        spaCy span with leading determinant removed
    """
    if span[0].pos_ == 'DET':
        # remove leading determinant and return resulting text
        return span[1:]
    
    return span


def get_spacy_preds(txt: str) -> List[spacy.tokens.span.Span]:
    """Return a list of spans corresponding to geographical location entities
    predicted by spaCy from a text.

    Input:
        txt: text to predict localities from.
    
    Output:
        List of spans corresponding to unique spaCy predicted locality entities
        Empty list if no spaCy predictions
    """
    # get spaCy locality predictions for txt
    spacy_preds = \
        [ent for ent in nlp(txt).ents if ent.label_ in SPACY_LOCALITIES]

    # keep only unique spaCy locality predictions
    spacy_preds = [remove_leading_determinant(pred) for pred in spacy_preds]
    loc_names = []
    unique_preds = []
    for pred in spacy_preds:
        if pred.text not in loc_names:
            loc_names.append(pred.text)
            unique_preds.append(pred)
    
    return unique_preds


def get_most_likely_region(location, geonames_result, geo_preds_regions) -> \
    Optional[Tuple[str, str, bool]]:
    """Using the top 50 geonames results for a location and a list of countries
    already identified by flashgeotext, get the most likely region/country of
    origin for this location.

    If no geonames search results, return None
    """
    if not geonames_result:
        return None, None

    # preferably return geonames result with exact string match, if possible
    if geonames_result[0].address == location:
        if geonames_result[0].country is None:
            # set country of origin to location itself, if location is not a
            # subregion to some country according to geonames
            return location, location
        else:
            return geonames_result[0].country, location

    # likely regions are regions that have already been identified by flashgeotext
    # as countries in the text, and have also been identified by geonames as
    # potential countries for this location
    #
    # sort likely regions with i, so we can pick the first geonames
    # result that overlaps with flashgeotext predicted regions
    #
    # country = country of origin for location, address = "canonical" name for
    # location from geonames
    likely_regions = sorted(
        [(res.country, res.address, i) for i, res in enumerate(geonames_result) \
            if res.country in geo_preds_regions],
            key=lambda x: x[2]
    )

    top_region_hit = geonames_result[0]

    if not likely_regions:
        # countries detected by flashgeotext in text doesn't correspond to top
        # result found by geonames, so just return the country + normalized
        # name of the top result
        if not top_region_hit.country:
            # no country, so treat location as a region
            return top_region_hit.address, top_region_hit.address
        
        return top_region_hit.country, top_region_hit.address
    
    else:
        # return first search result overlapping with regions predicted
        # by flashgeotext
        return likely_regions[0][0], likely_regions[0][1] 


def set_as_region_or_subregion(loc: str, regions: dict) -> None:
    """Assign a location (loc) as a subregion or a region using geonames.
    """
    region_names = list(regions.keys())

    geonames_result = get_cached_geonames_results(loc)

    most_likely_region, normalized_location_name = \
        get_most_likely_region(loc, geonames_result, region_names)
    
    if not most_likely_region:
        # treat location as a region, if no results from geonames
        regions[loc] = {'subregions': {}, 'found_as': [loc]}
    elif most_likely_region == normalized_location_name:
        # if normalized location name is same as most likely region, then
        # location was a region itself according to geonames
        regions[most_likely_region] = {'subregions': {}, 'found_as': [loc]}
    else:
        # assign location as a subregion to the most likely region
        if most_likely_region in regions:
            if normalized_location_name not in regions[most_likely_region]['subregions']:
                regions[most_likely_region]['subregions'][normalized_location_name] = \
                    [loc]
            else:
                regions[most_likely_region]['subregions'][normalized_location_name].append(loc)
        else:
            regions[most_likely_region] = {'subregions': {normalized_location_name: [loc]}, \
                                           'found_as': [most_likely_region]}


def get_regions_and_subregions(spacy_preds: List[spacy.tokens.span.Span]) -> \
    Dict[str, dict]:
    """For a list of spaCy entities corresponding to probable geographic
    locations, assign them to their likely origin regions or set them as
    their own region.
    
    Return a dictionary with keys as region names, and values as dictionaries
    of subregions for each region and a list of aliased names to this region.
    """
    regions = {}
    undetermined = []

    for pred in spacy_preds:
        geotext_pred = geotext.extract(pred.text)

        if geotext_pred['countries']:
            # geotext predicted this location as a country, so add it as
            # a region using the geotext normalized name as the region
            # key name in the dictionary
            regions[list(geotext_pred['countries'].keys())[0]] = \
                {'subregions': {}, 'found_as': [pred.text]}

        else:
            # for locations tagged as cities or not recognized at all by
            # flashgeotext, add these to 'undetermined' so we can link
            # them back to their respective regions or set them as
            # their own regions
            undetermined.append(pred.text)

    # start assigning undetermined regions back to their regions/countries,
    # or mark them as their own independent regions
    for loc in undetermined:
        set_as_region_or_subregion(loc, regions)

    return regions


def remove_taxonomic_entities(spacy_preds: List[spacy.tokens.span.Span],
                              txt: str) -> \
                                  None:
    """From a list of spaCy entities, remove any that are substrings of any
    TaxoNERD predicted taxonomic entities, or any which have taxonomic entities
    as substrings.
    
    Return None, mutating spacy_preds in place.
    """
    taxonomic_ents = taxonerd.find_in_text(txt)
    if taxonomic_ents.empty:
        return
    
    taxonomic_ents = taxonomic_ents['text'].tolist()

    preds_to_remove = []
    for pred in spacy_preds:
        # spacy entity is a substring of any predicted taxonomic entity, or
        # any predicted taxonomic entity is substring of spacy entity
        if any([pred.text in taxo or taxo in pred.text for \
            taxo in taxonomic_ents]):
            preds_to_remove.append(pred)

    for pred in preds_to_remove:
        spacy_preds.remove(pred)


def get_localities_string(locs: dict) -> str:
    """Return a string from some dictionary of predicted/normalized locations,
    in the form of "region 1 (subregion 1 | subregion 2 | ...); region 2 ..."
    """
    loc_strs = []
    for region in locs:
        loc_strs.append(
            f"{region} ({' | '.join(list(locs[region]['subregions'].keys()))})"
            )
    
    return '; '.join(loc_strs)


def predict_localities(txt: str) ->  Dict[str, dict]:
    """For a text, use flashgeotext + spaCy to predict countries and their
    associated cities/internal regions/etc.

    Input:
        txt: text to predict localities from

    Return:
        Dictionary with keys as regions and values as subregions
    """
    # get spaCy entities from text that correspond to geographical locations
    spacy_preds = get_spacy_preds(txt)

    # remove any location predictions that are actually just taxonomic names,
    # as those tend to get tagged as location entities by spacy
    remove_taxonomic_entities(spacy_preds, txt)

    # return the predicted locations as a dictionary of regions and their
    # subregions (ex: region = Australia, subregions = [Melbourne, Sydney])
    return get_regions_and_subregions(spacy_preds)

################################################################################

## Helper functions for normalizing recorded localities, so they're consistent
## with predicted localities
def clean_locality_string(loc: str) -> str:
    """Docstring goes here.
    """
    exceptions_dict = {
        '(pond (Mavlukeevskoe lake) within Tom River flood plain, Tomsk region, Western Siberia) Russia': \
            'Russia (Mavlukeevskoe lake in Tom River flood plain, Tomsk region, Western Siberia)',
            '(pond (Krotovo Lake) located near the village of Troitskoe, Novosibirsk region, Western Siberia) Russia': \
                'Russia (Krotovo Lake near the village of Troitskoe, Novosibirsk region, Western Siberia)'
    }

    if loc in exceptions_dict:
        return exceptions_dict[loc]

    loc = loc.strip()

    if loc[0] == '(':
        tmp = loc.split(')')
        loc = tmp[1].strip() + ' ' + tmp[0] + ')'

    return loc


def create_locality_dictionary(recorded_locs: str) -> dict:
    """For a string of recorded localities from a text, turn this string into
    the same format as returned by predict_localities.
    
    Ex: U.S. (Baltimore); Belarus ->
    
        {'U.S.': {'subregions': {'Baltimore': ['Baltimore']}, 
                  'found_as': ['U.S.']},
         'Belarus': {'subregions': {}, 'found_as': ['Belarus']}}
    """
    recorded_locs = [clean_locality_string(s) for s in recorded_locs.split(';')]

    locs_dict = {}

    for loc in recorded_locs:
        # extract region name away from parenthesized subregion text
        region = loc.split('(')[0].strip()

        # extract parenthesized subregion text
        subregions = re.search('(?<=\().+(?=\))', loc)
        if subregions:
            # this nasty list comprehension separates all comma/'|' separated
            # elements in the parentheses and returns an unnested list
            subregions = [item for l in [s.split(', ') for s in \
                subregions.group(0).split(' | ')] for item in l]
        else:
            subregions = []
        
        locs_dict[region] = {'subregions': {s: [s] for s in subregions},
                             'found_as': [region]}

    return locs_dict


def flashgeotext_normalize_recorded_localities(recorded_locs_dict: dict,
                                               recorded_locs) -> \
                                                   Tuple[dict, dict]:
    """Use flashgeotext to replace localities names in recorded_locs with
    their canonical geonames names, if possible.
    
    Return a tuple of recorded_locs_dict but with all location names replaced
    by normalized names from flashgeotext, and a dictionary of all undetected
    regions and their subregions
    """
    flashgeotext_preds = geotext.extract(recorded_locs)
    unnormalized_locs = {}
    normalized_locs = copy.deepcopy(recorded_locs_dict)

    for region in recorded_locs_dict:
        normalized_region = ''
        for country in flashgeotext_preds['countries']:
            if region in flashgeotext_preds['countries'][country]['found_as']:
                # set region key name in deep copied dictionary for normalized
                # recorded locations to the canonical name from flashgeotext
                normalized_locs[country] = normalized_locs.pop(region)
                normalized_region = country
                break

        if not normalized_region:
            # region not detected by flashgeotext, so add to unnormalized_locs
            # dictionary for normalization by geonames later
            unnormalized_locs[region] = \
                {'subregions': [s for s in recorded_locs_dict[region]['subregions']],
                 'normalized_region': False}
        else:
            # still add region to unnormalized locs dictionary, so we can keep track
            # of any subregions to this region that can't be normalized
            unnormalized_locs[normalized_region] = \
                {'subregions': [s for s in recorded_locs_dict[region]['subregions']],
                 'normalized_region': True}

        for subregion in recorded_locs_dict[region]['subregions']:
            for city in flashgeotext_preds['cities']:
                # replace entry for subregion in normalized dictionary with
                # canonical name from flashgeotext
                if subregion in flashgeotext_preds['cities'][city]['found_as']:
                    if normalized_region:
                        normalized_locs[normalized_region]['subregions'][city] = \
                            normalized_locs[normalized_region]['subregions'].pop(subregion)

                        # remove subregion from unnormalized_locs as it has been normalized
                        unnormalized_locs[normalized_region]['subregions'].remove(subregion)
                    else:
                        normalized_locs[region]['subregions'][city] = \
                            normalized_locs[region]['subregions'].pop(subregion)
                        
                        unnormalized_locs[region]['subregions'].remove(subregion)
                    
                    break

    return normalized_locs, unnormalized_locs


def geonames_normalize_recorded_localities(unnormalized_locs: Dict[str, List[str]],
                                           recorded_locs_dict: dict) -> None:
    """Look up recorded regions/subregions not detected by flashgeotext in geonames,
    and try to replace their entries in recorded_locs with their geonames canonical
    names.
    
    Modifies recorded_locs in-place.
    """
    unnormalized_locs_copy = copy.deepcopy(unnormalized_locs)
    for region in unnormalized_locs_copy:
        normalized_region = region  # replace with canonical geonames name, if possible

        if not unnormalized_locs[region]['normalized_region']:
            # get canonical region name from geonames since name wasn't normalized
            # by flashgeotext
            normalized_region = get_geonames_canonical_region(region)
            if normalized_region is None:
                normalized_region = region
            else:
                recorded_locs_dict[normalized_region] = \
                    recorded_locs_dict.pop(region)

                unnormalized_locs[normalized_region] = unnormalized_locs.pop(region)
                unnormalized_locs[normalized_region]['normalized_region'] = True
        
        for subregion in unnormalized_locs[normalized_region]['subregions']:
            normalized_subregion = \
                get_geonames_canonical_subregion(subregion, normalized_region)
            
            if normalized_subregion is not None:
                recorded_locs_dict[normalized_region]['subregions'][normalized_subregion] = \
                    recorded_locs_dict[normalized_region]['subregions'].pop(subregion)
                
                unnormalized_locs[normalized_region]['subregions'].remove(subregion)


def get_geonames_canonical_region(region: str) -> Optional[str]:
    """Return the canonical country name for the geonames result with an exact
    string match to region, or the top result otherwise.

    Return None if no geonames search results for region.
    """
    results = get_cached_geonames_results(region)
    if not results:
        return  # no geonames results, return None

    return results[0].address


def get_geonames_canonical_subregion(subregion: str, normalized_region: str) -> \
    Optional[str]:
    """Get the canonical name for some subregion of interest, choosing the geonames
    search result with normalized_region as its country, or the top result otherwise.

    Input:
        subregion: subregion recorded for some region

        normalized_region: geonames/flashgeonames-normalized name for the region
        to this subregion
    """
    results = get_cached_geonames_results(subregion)
    if not results:
        return  # no geonames results, return None

    # get results with country of origin as normalized_region
    region_results = [res for res in results if res.country == normalized_region]

    # get results in region_results with exact name match to subregion
    exact_result = [res for res in region_results if res.address == subregion]

    if exact_result:
        return exact_result[0].address
    elif region_results:
        return region_results[0].address
    
    return results[0].address


def normalize_recorded_localities(recorded_locs: str) -> str:
    """For a semi-colon separated list of regions and subregions recorded from a
    text, make the recorded location names consistent with predicted location names
    using flashgeotext and geonames.
    
    Ex: U.S. (Baltimore | Miami, Florida); Belarus ->
        United States (Baltimore | Miami | Florida); Belarus
    """
    # Create a dictionary of regions to subregions, formatted in the same way
    # predict_localities returns
    recorded_locs_dict = create_locality_dictionary(recorded_locs)

    # Normalize region/subregion names with flashgeotext, if possible, returning
    # a dictionary of undetected regions to undetected subregions, to try and
    # normalize with geonames
    recorded_locs_dict, unnormalized_locs = \
        flashgeotext_normalize_recorded_localities(
            recorded_locs_dict, recorded_locs
            )

    # Normalize regions/subregions not detected by flashgeotext with geonames,
    # by searching them up in geonames and using the canonical name of the
    # best geonames hit
    geonames_normalize_recorded_localities(unnormalized_locs, recorded_locs_dict)

    return recorded_locs_dict

################################################################################

if __name__ == '__main__':
    main()