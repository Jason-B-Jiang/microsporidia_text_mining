# -----------------------------------------------------------------------------
#
# Predict microsporidia polar tube coils + lengths from papers
#
# Jason Jiang - Created: 2022/05/20
#               Last edited: 2022/06/02
#
# Mideo Lab - Microsporidia text mining
#
#
# -----------------------------------------------------------------------------

## Imports

import spacy
from spacy.matcher import Matcher
import pandas as pd
import re
from pathlib import Path
from typing import List, Optional

################################################################################

## Set up spaCy model

nlp = spacy.load('en_core_web_md')
matcher = Matcher(nlp.vocab)

################################################################################

## Cache for texts that have already been processed by spacy
SPACY_CACHE = {}

################################################################################

## Define matcher for extracting polar tube coil data from text

PT_RANGE = r'(to|or|-|and)'  # let this appear one or more times

# 'l' of 'coil' sometimes gets messed up by ocr, so allow any lowercase char there
PT_COIL = r'([Cc]oi[a-z](s|ed)?|[Ss]pire(s|d)?|[Tt]urn[s]?|[Tt]wist(s|ed)?)'

# for excluding measurements in microns, as these aren't measuring number of polar
# tube coils
MICRON_TERMS = r'(μm|μ|mkm|um|mum|µm|mu|microns?)'

# NOTE: if a range of polar tube coils is found, the matcher will return both the
# full range match, and the partial range match
PT_COIL_DATA_PATTERN = [{'TEXT': '(', 'OP': '?'},  # allow polar tube coils to be parenthesized
                        {'POS': 'NUM'},
                        {'TEXT': {'REGEX': PT_RANGE}, 'OP': '?'},
                        {'POS': 'NUM', 'OP': '?'},
                        # exclude measurements in microns, as these aren't
                        # counting number of polar tube coils
                        # {'TEXT': {'REGEX': MICRON_TERMS}, 'OP': '!'},
                        # allow for bracket close to come before or after possible
                        # micron term
                        {'TEXT': ')', 'OP': '?'},
                        # allow for a max of 5 words between coil measurement and
                        # mention of coil term
                        # I tried using '*' operator, but didn't work in some cases
                        # {'TEXT': {'REGEX': '[a-z]+'}, 'OP': '*'},
                        # allow for parenthesized text to come after measurement
                        {'TEXT': '(', 'OP': '?'},
                        {'TEXT': {'REGEX': '[a-z0-9]+'}, 'OP': '?'},
                        {'TEXT': {'REGEX': '[a-z0-9]+'}, 'OP': '?'},
                        {'TEXT': {'REGEX': '[a-z0-9]+'}, 'OP': '?'},
                        {'TEXT': {'REGEX': '[a-z0-9]+'}, 'OP': '?'},
                        {'TEXT': {'REGEX': '[a-z0-9]+'}, 'OP': '?'},
                        {'TEXT': ')', 'OP': '?'},
                        {'TEXT': {'REGEX': PT_COIL}}]

matcher.add('pt_coil_data', [PT_COIL_DATA_PATTERN])

################################################################################

## Predicting polar tube coils

# Approach: extract sentences containing polar tube data from titles + abstracts,
# then extract polar tube coil data from these sentences

# Helper function for extracting polar tube coil data from sentences containg
# polar tube mentions
def extract_pt_coils(pt_sent: spacy.tokens.span.Span) -> str:
    """For a spaCy span representing a sentence containing polar tubule data,
    return a ' ||| ' separated string of polar tube coil measures extracted
    from the sentence.

    Input:
        pt_sents: spaCy span representing a sentence with polar tube data

    Return:
        ' ||| ' separated string of polar tube coil measures
        Empty string if no polar tube coil data is found
    """
    matches = matcher(pt_sent)

    if not matches:
        # no polar tube coil data was detected in sentence
        return ''

    # get longest span capturing the polar tube coil data
    # check match spans for overlap with other spans, and merge overlapping
    # spans into the longest possible span containing polar tube coil data
    curr_start = matches[0][1]
    curr_end = matches[0][2]
    for i in range(1, len(matches)):
        # current match doesn't overlap with previous match, so set this span
        # as current start and end
        if matches[i][2] > curr_end:
            curr_start = matches[i][1]
            curr_end = matches[i][2]

        elif matches[i][2] == curr_end and matches[i][1] < curr_start:
            # current match falls within previous match (i.e: is a substring of
            # the previous match), so remove from our matches
            curr_start = matches[i][1]
            matches[i - 1] = None

        elif matches[i][1] > curr_start:
            # current match is substring falling between start and end of previous
            # match, so remove from our matches
            matches[i] = None

    matches = list(filter(lambda s: s is not None, matches))

    # remove any matches with microns detected, as these probably aren't measuring
    # number of polar tube coils
    # ex: could be polar tube length, or size of microsporidia spores
    matches_text = \
        list(filter(
            lambda s: not re.search(MICRON_TERMS, s),
            [pt_sent[start : end].text for id, start, end in matches]
        ))
    
    return ' ||| '.join(matches_text)
        

# Function for extracting polar tube coil measurements from sentences with
# polar tube coil data
PT_REGEX = r'([Pp]olar ?)?([Ff]ilament|[Tt]ub(ul)?e)[s]?( |\.|,)'

def predict_polar_tube_coils(txt: str) -> Optional[str]:
    """Predict polar tube coil measurements from a text, by getting sentences
    containing polar tube data and extracting coil data from these sentences.

    Input:
        txt: Title + abstract of a microsporidia paper.
    """
    if txt in SPACY_CACHE:
        doc = SPACY_CACHE[txt]
    else:
        doc = nlp(txt)
        SPACY_CACHE[txt] = doc

    pt_sents = [sent for sent in doc.sents if re.search(PT_REGEX, sent.text)]

    if not pt_sents:
        # None if no sentences containing polar tube data are detected
        return None

    pt_coil_preds = list(map(extract_pt_coils, pt_sents))
    pt_coil_preds = list(filter(lambda s: s != '', pt_coil_preds))

    if pt_coil_preds:
        return ' ||| '.join(pt_coil_preds)
    else:
        return None

################################################################################

## Predict microsporidia polar tube coils from paper titles + abstracts

# Load in formatted dataframe of microsporidia species data, from
# src/1_format_data/3_misc_cleanup.R
microsp_data = pd.read_csv('../../data/manually_format_multi_species_papers.csv')

# Exclude species with >1 papers describing them (for now)
microsp_data = microsp_data[microsp_data['num_papers'] < 2]

# Add column for predicted polar tube coil measurements
microsp_data = microsp_data.assign(
    pred_pt_coil = lambda df: df['title_abstract'].map(
        lambda txt: predict_polar_tube_coils(txt)
    )
)

################################################################################

## Predicting polar tube length
PT_LENGTH_DATA_PATTERN = [{'LOWER': 'polar'},
                          {'TEXT': {'REGEX': '([Ff]ilament|[Tt]ub(ul)?e)[s]?'}},
                          # allow for 3 intervening tokens between polar tube
                          # mention and length measure
                          # ex: words (ex: "is"), punctuation, etc
                          {'TEXT': {'REGEX': '.+'}, 'OP': '?'},
                          {'TEXT': {'REGEX': '.+'}, 'OP': '?'},
                          {'TEXT': {'REGEX': '.+'}, 'OP': '?'},
                          {'POS': 'NUM'},
                          {'TEXT': {'REGEX': PT_RANGE}, 'OP': '?'},
                          {'POS': 'NUM', 'OP': '?'},
                          {'TEXT': {'REGEX': MICRON_TERMS}}]

# If polar tube length measure isn't properly spaced (ex: measurement + microns
# all mushed together, like "94.2±11.97μm"), allow numeric token w/ micron term
# inside it
PT_LENGTH_DATA_PATTERN_2 = [{'LOWER': 'polar'},
                          {'TEXT': {'REGEX': '([Ff]ilament|[Tt]ub(ul)?e)[s]?'}},
                          # allow for 3 intervening tokens between polar tube
                          # mention and length measure
                          # ex: words (ex: "is"), punctuation, etc
                          {'TEXT': {'REGEX': '.+'}, 'OP': '?'},
                          {'TEXT': {'REGEX': '.+'}, 'OP': '?'},
                          {'TEXT': {'REGEX': '.+'}, 'OP': '?'},
                          # in case polar tube length measure isn't properly spaced
                          # (ex: measurement + microns all mushed together, like
                          # "94.2±11.97μm", allow numeric token w/ micron term)
                          # TODO - rewrite this regex pattern using f string + format
                          {'POS': 'NUM', 'TEXT': {'REGEX': '(μm|μ|mkm|um|mum|µm|mu|microns?)'}}]

matcher.remove('pt_coil_data')
matcher.add('pt_length_data', [PT_LENGTH_DATA_PATTERN])
matcher.add('pt_length_data_2', [PT_LENGTH_DATA_PATTERN_2])


def extract_pt_length(pt_sent: spacy.tokens.span.Span) -> str:
    """For a spaCy span representing a sentence containing polar tubule data,
    return a ' ||| ' separated string of polar tube length measures extracted
    from the sentence.

    NOTE: most of this code is duplicated from extract_pt_coil. I should refactor
          my code to remove this code duplication.

    Input:
        pt_sents: spaCy span representing a sentence with polar tube data

    Return:
        ' ||| ' separated string of polar tube length measures
        Empty string if no polar tube coil length is found
    """
    matches = matcher(pt_sent)

    if not matches:
        # no polar tube coil data was detected in sentence
        return ''
    else:
        # if matches from first polar tube length pattern, keep only matches from
        # this pattern as more specific
        if [match for match in matches if nlp.vocab.strings[match[0]] == 'pt_length_data']:
            matches = list(filter(lambda match: nlp.vocab.strings[match[0]] == 'pt_length_data',
                                  matches))

    # get longest span capturing the polar tube coil data
    # check match spans for overlap with other spans, and merge overlapping
    # spans into the longest possible span containing polar tube coil data
    curr_start = matches[0][1]
    curr_end = matches[0][2]
    for i in range(1, len(matches)):
        # current match doesn't overlap with previous match, so set this span
        # as current start and end
        if matches[i][2] > curr_end:
            curr_start = matches[i][1]
            curr_end = matches[i][2]

        elif matches[i][2] == curr_end and matches[i][1] < curr_start:
            # current match falls within previous match (i.e: is a substring of
            # the previous match), so remove from our matches
            curr_start = matches[i][1]
            matches[i - 1] = None

        elif matches[i][1] > curr_start:
            # current match is substring falling between start and end of previous
            # match, so remove from our matches
            matches[i] = None

    matches = list(filter(lambda s: s is not None, matches))
    matches_text = [pt_sent[start : end].text for id, start, end in matches]
    
    return ' ||| '.join(matches_text)


def predict_polar_tube_length(txt: str) -> Optional[str]:
    """Predict polar tube length measurements from a text, getting sentences
    containing polar tube data and extracting polar tube length data if
    available.
    """
    if txt in SPACY_CACHE:
        doc = SPACY_CACHE[txt]
    else:
        doc = nlp(txt)
        SPACY_CACHE[txt] = doc

    pt_sents = [sent for sent in doc.sents if re.search(PT_REGEX, sent.text)]

    if not pt_sents:
        # None if no sentences containing polar tube data are detected
        return None

    pt_length_preds = [extract_pt_length(sent) for sent in pt_sents]
    pt_length_preds = [pred for pred in pt_length_preds if pred != '']

    if pt_length_preds:
        return ' ||| '.join(pt_length_preds)
    else:
        # return None instead of empty string, so we have NaNs in pandas dataframe
        # where no polar tube length predictions are made
        return None


# Add column for predicted polar tube length measurements
microsp_data = microsp_data.assign(
    pred_pt = lambda df: df['title_abstract'].map(
        lambda txt: predict_polar_tube_length(txt)
    )
)

################################################################################

# Write predictions to results folder
microsp_data[['species', 'title_abstract', 'pred_pt_coil', 'pt_coils_range',
              'pt_coils_avg', 'pred_pt', 'pt_max', 'pt_min', 'pt_avg']].to_csv(
    Path('../../results/microsp_pt_predictions.csv')
    )