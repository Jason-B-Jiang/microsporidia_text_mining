# -----------------------------------------------------------------------------
#
# Predict microsporidia sites of infection in hosts
#
# Jason Jiang - Created: Dec/05/2022
#               Last edited: Dec/05/2022
#
# Mideo Lab - Microsporidia text mining
#
# -----------------------------------------------------------------------------

import spacy
import scispacy
from scispacy.linking import EntityLinker
import pandas as pd

from typing import Union, List

###############################################################################

nlp = spacy.load('en_core_sci_sm')  # scispaCy large model

# add umls entity linker for all entities detected by scispaCy large model
nlp.add_pipe('scispacy_linker', config={'resolve_abbreviations': True,
                                        'linker_name': 'umls'})

###############################################################################

## Helper functions for the next part

def format_infection_sites(sites: Union[str, float]) -> List[str]:
    if isinstance(sites, float):
        return []

    return [s.split(' (')[0] for s in sites.split('; ')]


def get_n_sites_with_umls_entity(sites: List[str],
                                 ents: List[spacy.tokens.span.Span]) -> int:
    # site has umls entity if site is a substring of any captured entities
    # remove any captured entities from ents
    n_umls_entity = 0
    missing = []

    for site in sites:
        matches = {ent for ent in ents if site in ent.text or ent.text in site}
        ents = ents - matches
        n_umls_entity += len(matches) > 0

        if len(matches) == 0:
            missing.append(site)

    return n_umls_entity, missing

###############################################################################

n_infection_site = 0
n_infection_site_entities = 0
missing = []

infection_sites = \
    pd.read_csv('../../../data/microsp_infection_sites.csv')[
        ['title_abstract', 'infection_site_corrected']
    ]

for text, sites in zip(infection_sites['title_abstract'],
                      infection_sites['infection_site_corrected']):
    doc = nlp(text)
    sites = format_infection_sites(sites)
    ents = {ent for ent in doc.ents if ent._.kb_ents}

    n_infection_site += len(sites)
    x = get_n_sites_with_umls_entity(sites, ents)
    n_infection_site_entities += x[0]

    missing.append({'doc': doc, 'sites': sites, 'missing': x[1]})

missing = [m for m in missing if len(m['missing']) > 0]