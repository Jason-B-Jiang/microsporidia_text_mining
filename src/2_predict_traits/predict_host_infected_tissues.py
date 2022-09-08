# -----------------------------------------------------------------------------
#
# Predict microsporidia sites of infection in hosts
#
# Jason Jiang - Created: 2022/06/02
#               Last edited: 2022/07/26
#
# Mideo Lab - Microsporidia text mining
#
# -----------------------------------------------------------------------------

import spacy
import scispacy
from spacy.matcher import Matcher
from typing import List, Tuple, Dict
from scispacy.linking import EntityLinker
import re
import pandas as pd
from pathlib import Path

################################################################################

## Model initialization

# Alternatively: use large model to tag all entities, then link to UMLS
nlp = spacy.load("en_core_sci_lg")  # use large model for optimal entity tagging

# Add entity linking from the Unified Medical Language System, for getting
# normalizing tissue/organ/etc names w/ canonical UMLS names
#
# Only add entity linking for the en_core_sci_lg pipeline, due to memory
# constraints
nlp.add_pipe("scispacy_linker",
             config={"resolve_abbreviations": True, "linker_name": "umls"})

linker = nlp.get_pipe("scispacy_linker")

################################################################################

def main() -> None:
    microsp_data = pd.read_csv('../../data/manually_format_multi_species_papers.csv')

    # Fill missing values in these columns with empty strings
    microsp_data[['infection_site', 'hosts_natural', 'hosts_experimental', 'abstract', 'title_abstract']] = \
        microsp_data[['infection_site', 'hosts_natural', 'hosts_experimental', 'abstract', 'title_abstract']].fillna('')

    # clean recorded infection sites
    microsp_data['infection_site_formatted'] = microsp_data.apply(
        lambda df: clean_recorded_infection_sites(df.infection_site, df.hosts_natural,
                                                  df.hosts_experimental),
            axis=1)

    # normalize recorded infection sites with umls names (if possible) and get
    # predicted infection sites
    microsp_data[['infection_site_normalized', 'pred_infection_site', 'raw_predictions']] = \
        [predict_and_normalize_infection_sites(txt, sites_formatted) for \
            txt, sites_formatted in \
                zip(microsp_data.abstract, microsp_data.infection_site_formatted)]
   
    microsp_data['sites_not_in_text'] = \
        microsp_data.apply(lambda df: sites_not_in_text(df.infection_site_formatted,
            df.title_abstract), axis = 1)
    
    microsp_data[['species', 'num_papers', 'paper_title', 'abstract', 'infection_site',
                  'infection_site_formatted', 'raw_predictions', 'infection_site_normalized',
                  'pred_infection_site', 'sites_not_in_text']].to_csv(
                      Path('../../results/microsp_infection_site_predictions.csv')
                  )

################################################################################

## HELPER CODE + FUNCTIONS

################################################################################

## Cache texts as we process them with spaCy, to speed up code (potentially)
CACHED_TEXT = {}

def get_cached_text(txt: str) -> spacy.tokens.doc.Doc:
    """Retrieve cached results for some string, txt, already processed by
    either the bionlp13cg_md or en_core_sci_lg spaCy model.

    Inputs:
        txt: string to retrieve cached spaCy results for

        spacy_large: bool indicating if we want results from en_core_sci_lg
        model. If False, then get cached results from bionlp13cg_md model
    """
    if not txt in CACHED_TEXT:
        CACHED_TEXT[txt] = nlp(txt)
    
    return CACHED_TEXT[txt]

################################################################################

## Cleaning up recorded sites of infection

# Manually parsed from inspecting 'Site of Infection' column
# "Unimportant" parenthesized information following recorded infection sites, to
# remove from recorded infection sites
IRRELEVANT_PARENTHESIZED_INFO =\
    ['except', 'main', 'site', 'larva', 'adult', 'male', 'female', '\?', 'type',
    'at first', 'after', 'rarely', 'in ', 'spores', 'secondarily', 'between',
    'in advanced cases', 'prominent', 'close', '1', '2', 'chickens', 'of ',
    'does not', 'wall', 'low infection', 'all hosts', 'juvenile', 'embryo',
    'near', 'in one host individual', 'vertical', 'colonies', 'most organs',
    'similar tissues infected for both hosts', 'heavily infected', 'havily infected',
    'most infected' 'free spores', 'primarily', 'of Corethra (Savomyia) plumicornis',
    'anterior end', 'posterior end', 'most heavily infected', 'including',
    'of homo sapiens', 'of athymic mice']

# Parenthesized information following recorded infection sites that IS informative,
# as it tells us more information about where the Microsporidia infects, so keep
# this info
EXCLUSIONS = \
    ['in lamina propria through muscularis mucosae into submucosa of small intestine tract',
    'mainly in duodenum']


def get_abbrev_species_name(host: str) -> str:
    """For a taxonomic species name, host, return its abbreviated species name.
    Ex: Nosema bombycis -> N. bombycis.
    Ex: Aedes punctor punctor -> A. p. punctor

    Keep already abbreviated names as is
    Ex: N. bombycis -> N. bombycis (unchanged)
    """
    if re.search("[A-Z]\. ", host):
        # host name already abbreviated, return as is
        return host
    
    host = host.split(' ')
    abbrevs = [s[0] + '.' for s in host[:len(host) - 1]]

    return ' '.join(abbrevs + [host[len(host) - 1]])


def get_host_names_and_abbrevs(hosts_natural, hosts_experimental) -> List[str]:
    """For semi-colon separated strings of recorded natural and experimental
    hosts for a microsporidia species, return a list of the abbreviated names
    for all the species.

    Inputs:
        hosts_natural: semi-colon separated string of natural infection hosts

        hosts_experimental: semi-colon separated string of experimental
        infection hosts
    """
    hosts_natural = [h.split('(')[0].strip() for h in \
        hosts_natural.split('; ')]

    hosts_experimental = [h.split('(')[0].strip() for h \
        in hosts_experimental.split('; ')]

    all_hosts = hosts_natural + hosts_experimental
    all_hosts.extend([get_abbrev_species_name(h) for h in all_hosts])

    return(list(filter(lambda s: s != '', all_hosts)))


def should_remove_subsite(subsite: str, all_host_names: List[str]) -> bool:
    """Determine whether to remove a subsite (i.e: parenthesized text) associated
    with some recorded infection site.

    Return True (should remove the recorded subsite) if it corresponds to
    an irrelevant term that doesn't have to do with infection sites, or if it
    corresponds to some taxonomic species name.

    Inputs:
        subsite: string of the parenthesized text/subsite info associated with
        a recorded Microsporidia infection site

        all_host_names: a list of all host names (full + abbreviated names) for
        a Microsporidia species, to sus out subsite info that corresponds to
        taxonomic names (and should be removed)
    """
    return any([term in subsite for term in \
        IRRELEVANT_PARENTHESIZED_INFO + all_host_names]) and \
        not any([term in subsite for term in EXCLUSIONS])


def clean_recorded_infection_sites(sites: str, hosts_natural: str,
                                   hosts_experimental: str) -> str:
    """Format recorded sites of infection by ensuring parenthesized text
    are indicating subcellular/subtissue locations only (i.e: remove
    parenthesized text that corresponds to taxonomic host names, and
    other irrelevant info that doesn't inform us more about infection
    sites).

    Return a semi-colon separated string of all infection sites and
    their relevant subsites, all separated by semi-colons.

    Inputs:
        sites: semi-colon separated string of recorded infection sites for
        some Microsporidia species

        hosts_natural: semi-colon separated string of naturally infected
        hosts for some Microsporidia species

        hosts_experimental: semi-colon separated string of experimentally
        infected hosts for some Microsporidia species
    """
    # replace('(1;2)', '') fixes a corner case in a single entry where a
    # semi-colon was accidentally used to separate subsite entries
    # (semi-colons should only separate full infection site entries)
    sites = [s.strip() for s in sites.replace('(1;2)', '').split(';')]
    sites_formatted = []

    all_host_names = get_host_names_and_abbrevs(hosts_natural, hosts_experimental)

    for site in sites:
        # extract parenthesized info and split into subentries w/ ', '
        # strip away parenthesized text from main entry
        # remove subentries which have anything in irrelevant info as substring
        #   (or any entries in host species)
        # extend entries + subentries to sites_formatted
        if '(' in site:
            # subsites are recorded for site, indicated by parenthesized text
            subsites = \
                [s.strip().strip(')') for s in \
                    re.search("(?<=\().+\)?", site).group(0).split(',')]
        else:
            subsites = []
        
        # only keep informative subsites for each infection site
        subsites = [s for s in subsites if not \
            should_remove_subsite(s, all_host_names)]
        
        # add sites + subsites together into one list
        sites_formatted.append(site.split('(')[0].strip())
        sites_formatted.extend(subsites)
    
    # return a single string containing all infection sites + subsites,
    # separated by semi-colon
    return '; '.join(sites_formatted)


def sites_not_in_text(sites_formatted: str, text: str) -> str:
    """Return semi-colon separated list of all recorded sites not found in the
    text it was extracted from.
    
    Use the 'cleaned' strings for infection sites, where all irrelevant
    parenthesized info has been stripped away.
    """  
    return '; '.join(
        [site for site in sites_formatted.split('; ') if \
            site.lower() not in text.lower()]
    )

################################################################################

## Predicting sites of infection from paper abstracts

# Base words in a sentence indicates it has info about Microsporidia infection
# sites
INFECTION_LEMMAS = ['find', 'parasitize', 'infect', 'infeet', 'describe',
                    'localize', 'invade', 'appear', 'parasite', 'infection',
                    'appear', 'invasion', 'occur']

infection_matcher = Matcher(nlp.vocab)
infection_matcher.add('infection_site',
                      [[{'LEMMA': {'IN': INFECTION_LEMMAS}}]])

# Words that correspond to Microsporidia anatomy, and not host infection sites,
# and so shouldn't be part of predicted infection sites of a Microsporidia species
MICROSPORIDIA_PARTS = ['polar tube', 'polar filament', 'sporoblast', 'spore'
                       'meront', 'meronts', 'sporoplasm', 'sporophorous vesicles',
                       'sporont', 'sporonts' 'polaroplast', 'anchoring disc',
                       'lamellar body', 'anchoring disk', 'endospore', 'exospore',
                       'posterior vacuole', 'sporoblasts', 'meiospores', 'meiospore',
                       'macrospore', 'macrospores', 'microspore', 'microspores',
                       'basal', 'spores', 'schizogony', 'sporogony']

# IDs of UMLS entries that we are interested in
RELEVANT_UMLS_TYPES = ('T017', 'T018', 'T022', 'T023', 'T024', 'T025', 'T029',
                       'T030', 'T031')


def is_overlapping_spans(s1: spacy.tokens.span.Span, s2: spacy.tokens.span.Span) -> \
    bool:
    """Return True if two spaCy spans have overlapping boundaries in a document.
    """
    if len(set(range(s1.start, s1.end + 1)) & \
        set(range(s2.start, s2.end + 1))) > 0:
        return True
    
    return False


def get_overlapping_entity(site: spacy.tokens.span.Span, \
    ents: List[spacy.tokens.span.Span]) -> spacy.tokens.span.Span:
    """Check if some span has any overlap (i.e: is a substring of) any
    entity in some list of entity spans, ents.

    Inputs:
        site: span to check for overlaps with entity list
        ents: list of spans corresponding to spaCy entities of interest
    """
    for ent in ents:
        if is_overlapping_spans(ent, site):
            return ent
    
    return # return None if no overlapping entity found


def get_site_spans(doc: spacy.tokens.doc.Doc) -> List[spacy.tokens.span.Span]:
    """For a semi-colon separated string of recorded infection sites,
    return the corresponding list of spans for each item in this
    semi-colon separated string

    Input:
        doc: semi-colon separated string of recorded infection sites that
        has been processed by a spaCy pipeline
    """
    semicolon_posns = [i for i, tok in enumerate(doc) if tok.text == ';']
    curr_posn = 0
    sites = []

    for i in semicolon_posns:
        sites.append(doc[curr_posn : i])
        curr_posn = i + 1

    if semicolon_posns:
        return sites + [doc[semicolon_posns[-1] + 1:]]

    return [doc[:]]  # return doc as a span and 'as is' if no semicolons


def get_recorded_sites_dict(recorded_infection_sites: str) -> dict:
    """Docstring goes here.
    """
    doc = nlp(recorded_infection_sites)
    sites_dict = {}

    # get spans for all recorded infection sites
    site_spans = get_site_spans(doc)
    for span in site_spans:
        if not any([is_overlapping_spans(span, ent) for ent in doc.ents]):
            # recorded site wasn't detected as entity by scispacy,
            # so make sure we still include it in sites_dict
            #
            # no umls entries, so set normalized name as original text
            sites_dict[span.text] = {
                'umls_entries': [],
                'normalized_name': span.text
                }

    # don't need to check if entities in recorded sites are not Microsporidia
    # parts or if they have valid UMLS entries, as we are confident we recorded
    # relevant terms as the infection sites
    #
    # possible "uninformative" entities to exclude:
    # cell, cells, tissue, tissues (use lemmas for singulars)
    for ent in doc.ents:
        sites_dict[ent.text] = {
            'umls_entries': ent._.kb_ents,
            'normalized_name': None
        }

    return sites_dict


def has_valid_umls_entries(ent: spacy.tokens.span.Span) -> bool:
    """Docstring goes here.
    """
    for i in range(len(ent._.kb_ents)):
        concept_ids = \
            linker.kb.cui_to_entity[ent._.kb_ents[i][0]].types

        if not any([id in RELEVANT_UMLS_TYPES for id in concept_ids]):
            ent._.kb_ents[i] = None

    ent._.kb_ents = [ent for ent in ent._.kb_ents if ent is not None \
        and ent[1] >= 0.8]

    return len(ent._.kb_ents) > 0


def get_top_overlapping_umls_entry(umls_1: List[Tuple[str, float]],
                                   umls_2: List[Tuple[str, float]]) -> Tuple[str, float]:
    """Docstring goes here.
    """
    overlapping_umls_terms = [umls[0] for umls in umls_1 if umls[0] in \
        [u[0] for u in umls_2]]

    umls_1 = [umls for umls in umls_1 if umls[0] in overlapping_umls_terms]
    umls_2 = [umls for umls in umls_2 if umls[0] in overlapping_umls_terms]

    # sanity check for myself
    assert len(umls_1) == len(umls_2)

    top_avg_umls_confidence = 0.0
    top_umls_term = ""

    for i in range(len(umls_1)):
        curr_avg_confidence = (umls_1[i][1] + umls_2[i][1]) / 2
        if curr_avg_confidence > top_avg_umls_confidence:
            top_avg_umls_confidence = curr_avg_confidence
            top_umls_term = umls_1[i][0]

    if top_umls_term:
        return linker.kb.cui_to_entity[top_umls_term].canonical_name, \
            top_avg_umls_confidence
    
    return top_umls_term, top_avg_umls_confidence


def get_normalized_site_name(recorded_site: str, umls_entries: List[Tuple[str, float]],
                             pred_sites: Dict[str, dict]) -> str:
    """Docstring goes here.
    """
    normalized_name = recorded_site
    top_pred = ""

    # top average confidence in overlapping UMLS entry between recorded_site and some
    # predicted site
    top_avg_umls_confidence = 0.0 

    for pred in pred_sites:
        # only consider predicted sites that have not been normalized yet
        if pred_sites[pred]['normalized_name'] is None:
            if recorded_site in pred:
                # get top umls entry name for predicted site that has
                # recorded site as a substring, to set as normalized name
                normalized_name = \
                    linker.kb.cui_to_entity[
                        pred_sites[pred]['umls_entries'][0][0]
                    ].canonical_name

                top_pred = pred
                break
            
            curr_top_umls_entry, curr_avg_umls_confidence = \
                get_top_overlapping_umls_entry(umls_entries, pred_sites[pred]['umls_entries'])

            # if no overlapping umls entries were found between recorded and
            # predicted site, returned curr_avg_umls_confidence will be zero,
            # so nothing will happen
            if curr_avg_umls_confidence > top_avg_umls_confidence:
                top_avg_umls_confidence = curr_avg_umls_confidence
                normalized_name = curr_top_umls_entry
                top_pred = pred
        
    if top_pred:
        pred_sites[top_pred]['normalized_name'] = normalized_name

    return normalized_name


def resolve_normalized_names(pred_sites: dict, recorded_sites: str) -> \
    Tuple[str, str]:
    """Get UMLS normalized names for predicted sites + recorded sites.
    """
    # Process recorded infection sites with spaCy, and extract relevant entities
    recorded_sites = get_recorded_sites_dict(recorded_sites)

    for site in recorded_sites:
        if recorded_sites[site]['normalized_name'] is not None:
            # recorded site was not detected as entity, so skip it
            continue
        
        # get dictionary key from predicted sites that most likely
        # corresponds to this recorded site
        normalized_name = \
            get_normalized_site_name(site, recorded_sites[site]['umls_entries'],
                                     pred_sites)
        
        recorded_sites[site]['normalized_name'] = normalized_name

    for pred in pred_sites:
        # if predicted site didn't have corresponding recorded site, set
        # normalized name as top umls entry
        if pred_sites[pred]['normalized_name'] is None:
            pred_sites[pred]['normalized_name'] = \
                linker.kb.cui_to_entity[pred_sites[pred]['umls_entries'][0][0]].canonical_name

    return '; '.join([recorded_sites[site]['normalized_name'] for site in recorded_sites]), \
        '; '.join([pred_sites[site]['normalized_name'] for site in pred_sites]), \
            '; '.join([site for site in pred_sites])


def predict_and_normalize_infection_sites(txt: str, recorded_infection_sites: str) \
    -> Tuple[str, str]:
    """Docstring goes here.
    """
    txt = nlp(txt)
    infection_sents = [sent for sent in txt.sents if infection_matcher(sent)]

    # note: some relevant entities that get flagged but doesn't have
    # UMLS entries (ex: gastric caeca, oenocytes)
    pred_infection_sites = {}
    for sent in infection_sents:
        # collect entities from sentence that are possible Microsporidia infection
        # sites
        possible_infection_sites = \
            [ent for ent in sent.ents if not \
                any([part in ent.text.lower() for part in MICROSPORIDIA_PARTS]) and \
                    has_valid_umls_entries(ent)]

        for site in possible_infection_sites:
            pred_infection_sites[site.text] = {
                'umls_entries': site._.kb_ents,
                'normalized_name': None
            }
    
    # don't change umls name stuff betweeen observe and predicted, identify original
    # text where sites were inferred from
    #
    # get verbatim text from abstracts, does verbatim text match extracted entity
    #
    # mention subjectivity in making dataset for harmonizing entries
    # verbatim text may or may not be recorded
    #
    # allow mismatches b/c of this and just discuss in discusson
    # or, manually pull out verbatim text (or algorithmically)
    #
    # 
    return resolve_normalized_names(pred_infection_sites, recorded_infection_sites)

################################################################################

if __name__ == '__main__':
    main()