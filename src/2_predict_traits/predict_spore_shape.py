# -----------------------------------------------------------------------------
#
# Predict microsporidia spore shape
#
# Jason Jiang - Created: 2022/07/13
#               Last edited: 2022/07/20
#
# Mideo Lab - Microsporidia text mining
#
# -----------------------------------------------------------------------------

import spacy
from spacy.matcher import PhraseMatcher, Matcher
import pandas as pd
from pathlib import Path
from typing import List, Tuple

################################################################################

## Global variables

# Note: a lot of these variables are reused from predict_spore_nucleus_count.py

nlp = spacy.load('en_core_web_md')

# matcher for microsporidia shape descriptors
# these are terms that are commonly used in Microsporidia literature that I'm
# calling off the top of my head
shape_terms = ('oval', 'ovoid', 'round', 'pyriform', 'ovocylindrical',
               'spherical', 'ellipsoidal', 'ellipsoid', 'rod-shaped', 'rod')

shape_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
shape_matcher.add('shape', [nlp(term) for term in shape_terms])

# matcher for different types of mature microsporidia spores
# these spores are typically named [prefix]spore, ex: 'meiospores', 'macrospores'
# or, they're just called 'spores'
#
# make sure we don't match on 'exospore' or 'endospore', as these describe parts
# of microsporidia anatomy and not the spores themselves
#
# (same code from predict_spore_nucleus_count.py)
spore_matcher = Matcher(nlp.vocab)
spore_matcher.add('spore',  [[{'TEXT': {'REGEX': '(?<!(xo|do))[Ss]pore'}}]])

################################################################################

def main() -> None:
    microsp_data = pd.read_csv(
        '../../data/manually_format_multi_species_papers.csv'
        )

    microsp_data = microsp_data.assign(
        pred_shape = lambda df: get_spore_shape_string(
            predict_spore_shapes(df.title_abstract)
        )
    )

    microsp_data.to_csv(Path('../../results/microsp_spore_shape_predictions.csv'))

################################################################################

## Helper functions for predicting spore shapes + associated spore types from
## texts

def get_match_spans(matches: List[Tuple[int]], sent: spacy.tokens.span.Span,
                    doc: spacy.tokens.doc.Doc) -> \
    List[spacy.tokens.span.Span]:
    """Docstring goes here.
    """
    match_spans = []

    if doc:
        for match in matches:
            match_spans.append(doc[match[1] : match[2]])
    else:
        for match in matches:
            match_spans.append(sent[match[1] : match[2]])

    return match_spans


def match_shapes_to_spores(spore_spans: List[spacy.tokens.span.Span],
                           shape_spans: List[spacy.tokens.span.Span]) -> \
                               List[Tuple[str, str]]:
    """Match every spore shape to its corresponding spore type.
    Each spore type should only be associated with a single shape, but
    each shape may apply for multiple spores.

    Return a list of tuples of (spore type, spore shape).
    """
    if len(shape_spans) == 0:
        return []

    shapes_to_spores = []

    for spore in spore_spans:
        closest_shape = get_closest_shape(spore, shape_spans)
        shapes_to_spores.append((spore.lemma_, closest_shape.text))
    
    return shapes_to_spores


def get_closest_shape(spore: Tuple[str, int, int],
                      shape_spans: List[spacy.tokens.span.Span]) -> \
                          Tuple[str, int, int]:
    """Get closest shape to a spore type mentioned in a text, using
    start and stop coordinates of the spans for spore types/shapes
    to get closest matches.

    Break ties for shapes equidistant to a spore type by choosing the
    shape that comes before the spore in the text.
    """
    return sorted(shape_spans, key=lambda x: get_span_distance(spore, x))[0]


def get_span_distance(s1: spacy.tokens.span.Span,
                      s2: spacy.tokens.span.Span) -> int:
    """Return the number of tokens that two spans are separated by.
    Return -1 is s1 and s2 are the same span (i.e: there are no tokens
    separating the spans)
    """
    # s1 comes after s2
    if s1.start > s2.start:
        return s1.start - s2.end
    
    # s1 comes before s2
    return s2.start - s1.end


def get_spore_shape_string(spore_shapes: List[Tuple[str, str]]) -> str:
    """Docstring goes here.
    """
    s = []
    for tup in spore_shapes:
        if tup[0] == 'spores' or tup[0] == 'spore':
            s.append(f"{tup[1]} (normal spore)")
        else:
            s.append(f"{tup[1]} ({tup[0]})")
    
    return '; '.join(s)


def predict_spore_shapes(txt: str) -> str:
    """Predict shapes for each type of spore mentioned in some string, txt.
    Return a string in the form of "{shape} ({spore class name}); ..."

    Input:
        txt: abstract or full text for a Microsporidia paper
    """
    doc = nlp(txt)

    # get sentences from text with possible information about spores
    # (and thus spore shapes)
    # make a dictionary, with keys as spore sentences as values as
    # a dictionary of spore shapes + spore types in each sentence
    spore_sents = {sent: {'spore_types': spore_matcher(sent),
                          'spore_shapes': shape_matcher(sent)} \
                              for sent in doc.sents if spore_matcher(sent)}

    spore_shapes = []  # list of tuples, (spore name, spore shape)
    for sent in spore_sents:
        spore_spans = get_match_spans(spore_sents[sent]['spore_types'], sent, None)
        shape_spans = get_match_spans(spore_sents[sent]['spore_shapes'], sent, doc)

        spore_shapes.extend(match_shapes_to_spores(spore_spans, shape_spans))
    
    return spore_shapes

################################################################################

if __name__ == '__main__':
    main()
