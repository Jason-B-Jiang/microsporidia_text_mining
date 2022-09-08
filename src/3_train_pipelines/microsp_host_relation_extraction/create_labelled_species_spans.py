# -----------------------------------------------------------------------------
#
# Create labelled microsporidia species + host spans for NER training
#
# Jason Jiang - Created: 2022/08/01
#               Last edited: 2022/08/26
#
# Mideo Lab - Microsporidia text mining
#
# -----------------------------------------------------------------------------

import pandas as pd
from typing import List, Dict
import re
import spacy
from spacy.tokens import DocBin
import random
from pathlib import Path

################################################################################

## Global variables

random.seed(42)  # for consistent train/test splits
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.1

# load in blank pipeline, as we only want tokenization
nlp = spacy.blank('en')

################################################################################

def main() -> None:
    microsp_host_names = pd.read_csv('../../../results/TEMP_naive_name_preds.csv')

    # only select columns of interest for generating train/valid/test data split
    # for training spaCy models
    microsp_host_names = microsp_host_names[
        ['species', 'title_abstract', 'microsp_in_text', 'hosts_in_text']
    ]

    data_splits = get_labelled_documents(microsp_host_names)

    for split in data_splits:
        # save training, validation and testing data for each split as spaCy
        # 3 binary files (i.e: .spacy files)
        write_to_spacy_file(data_splits[split], split)

################################################################################

def get_labelled_documents(microsp_host_names: pd.core.frame.DataFrame) -> \
    Dict[str, List[spacy.tokens.doc.Doc]]:
    """Return a dictionary with keys as data split type (i.e: 'train', 'valid',
    'split') and values as spaCy documents (microsporidia paper abstracts) with
    microsporidia and host species names set as entities in the document.

    Arguments:
        microsp_host_names: dataframe of microsporidia paper abstracts and the
        microsporidia/host species names that are found in them

    """
    labelled_docs = []
    species_entries = []

    for i in range(len(microsp_host_names)):
        title_abstract = nlp(microsp_host_names['title_abstract'][i])
        microsp = get_species_list(microsp_host_names, 'microsp_in_text', i)
        hosts = get_species_list(microsp_host_names, 'hosts_in_text', i)

        # get spans of microsporidia and host species names for the spaCy
        # document of the associated paper title + abstract
        microsp_spans = get_spans_from_doc(microsp, title_abstract, 'MICROSPORIDIA')
        host_spans = get_spans_from_doc(hosts, title_abstract, 'HOST')

        # resolve microsporidia and host spans that are overlapping by removing
        # the overlapping spans, and set them as entities to document
        #
        # ex: microsporidia + host species have the same abbreviated names
        title_abstract.ents = resolve_overlapping_spans(microsp_spans, host_spans)

        # add this labelled document to our accumulating list of labelled documents
        labelled_docs.append(title_abstract)
        species_entries.append(microsp_host_names['species'][i])

    # separate labelled documents into train/valid/test split
    return get_train_valid_test_split(labelled_docs)


def get_species_list(microsp_host_names: pd.core.frame.DataFrame, col_name: str,
                     i: int) -> List[str]:
    """Docstring goes here.
    """
    species = microsp_host_names[col_name][i]

    if pd.isna(species):
        # nan (i.e: no recorded species in that column), so return empty list
        return []
    
    species = species.split('; ')

    # also add abbreviated names for each species to the list, and only
    # keep unique species names
    return list(set(species + [get_abbreviated_species_name(sp) for sp in species]))


def get_abbreviated_species_name(species: str) -> str:
    """Docstring goes here.
    """
    species = species.split(' ')
    if len(species) == 1:
        # species name is a single word, no abbreviation possible
        return species[0]
    
    return ' '.join([s[0] + '.' for s in species[:len(species) - 1]]) + \
        ' ' + species[-1]


def get_spans_from_doc(names: List[str], doc: spacy.tokens.doc.Doc, label: str) -> \
    List[spacy.tokens.span.Span]:
    """Docstring goes here.
    """
    match_spans = []

    for name in names:
        # find literal matches for species names in the document
        # make all text lowercase so string searches are case-insensitive
        for match in re.finditer(re.escape(name.lower()), doc.text.lower()):
            start, end = match.span()
            span = doc.char_span(start, end, label=label)

            if span is not None:
                match_spans.append(span)
    
    return match_spans


def resolve_overlapping_spans(spans_1: List[spacy.tokens.span.Span],
                             spans_2: List[spacy.tokens.span.Span]) -> \
                                List[spacy.tokens.span.Span]:
    """Docstring goes here.
    """
    spans_1_overlaps = [span for span in spans_1 if \
        any([is_overlapping_span(span, other) for other in spans_2])]

    spans_2_overlaps = [span for span in spans_2 if \
        any([is_overlapping_span(span, other) for other in spans_1])]

    [spans_1.remove(span) for span in spans_1_overlaps]
    [spans_2.remove(span) for span in spans_2_overlaps]

    return spans_1 + spans_2


def is_overlapping_span(span_1: spacy.tokens.span.Span,
                        span_2: spacy.tokens.span.Span) ->\
    bool:
    """Docstring goes here.
    """
    if len(set(range(span_1.start, span_1.end)) & \
        set(range(span_2.start, span_2.end))) > 0:
        return True
    
    return False


def get_train_valid_test_split(labelled_docs: List[spacy.tokens.doc.Doc]) -> \
    Dict[str, List[spacy.tokens.doc.Doc]]:
    """Docstring goes here.
    """
    random.shuffle(labelled_docs)
    train_idx = int(round(len(labelled_docs) * TRAIN_SPLIT))
    valid_idx = train_idx + int(round(len(labelled_docs) * VALID_SPLIT))

    return {'train': labelled_docs[:train_idx],
            'valid': labelled_docs[train_idx:valid_idx],
            'test': labelled_docs[valid_idx:]}


def write_to_spacy_file(labelled_docs: List[spacy.tokens.doc.Doc],
                        name: str) -> None:
    """For a list of spaCy documents labelled with microsporidia/host species
    entities in their texts, write these documents into a .spacy model training
    file.
    
    Arguments:
        labelled_docs: list of spaCy documents (paper abstracts) with microsporidia
        and host species names as named entities

        split_name: name to give to .spacy file for this collection of documents

    """
    db = DocBin(docs=labelled_docs)
    db.to_disk(Path(f"{name}.spacy"))

################################################################################

if __name__ == '__main__':
    main()