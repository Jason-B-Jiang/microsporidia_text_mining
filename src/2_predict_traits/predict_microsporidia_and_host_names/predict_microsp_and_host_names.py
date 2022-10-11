# -----------------------------------------------------------------------------
#
# Predict microsporidia species names + hosts from paper titles + abstracts
#
# Jason Jiang - Created: 2022/05/19
#               Last edited: 2022/10/11
#
# Mideo Lab - Microsporidia text mining
#
#
# -----------------------------------------------------------------------------

## Imports

import re
import pandas as pd
from taxonerd import TaxoNERD
import spacy
from spacy.matcher import Matcher
from typing import Tuple, Union, List, Dict, Set
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import copy

################################################################################

## Initialize language models + other global variables

taxonerd = TaxoNERD(model="en_ner_eco_biobert",  # use more accurate model
                    prefer_gpu=False,
                    with_abbrev=True)

nlp = spacy.load('en_core_web_md')

try:  # get cached microsporidia/host predictions for texts, if exists
    with open('./cached_microsp_host_preds.pickle', 'rb') as p:
        PREDICTIONS_CACHE = pickle.load(p)
except FileNotFoundError:
    PREDICTIONS_CACHE = {}

EXCLUDED_VERBS = {'propose', 'name', 'have', 'ribosomal', 'know', 'suggest'}

DOC_CACHE = {}

################################################################################

## Define regex patterns for detecting new Microsporidia species

# all microsporidia genus names from NCBI
MICROSP_GENUSES = \
    pd.read_csv('../../../data/microsp_genuses/microsporidia_genuses.tsv',
        sep='\t')['name'].tolist()

MICROSP_GENUSES = f"({'|'.join(MICROSP_GENUSES)})"

# conventional ways of indicating new species
# ex: sp. n, nov. sp., sp. nov., sp. n.
NEW_SPECIES_INDICATORS = \
    r' ([Nn](OV|ov)?(\.|,) (SP|sp)(\.|,)|(SP|sp)(\.|,) [Nn](OV|ov)?(\.|,))'

################################################################################

def main() -> None:
    microsp_and_host_names = pd.read_csv('../../../data/formatted_host_microsp_names.csv')

    # assign new columns for predicted microsporidia + host species names from
    # texts
    microsp_and_host_names[['pred_microsp', 'pred_hosts']] = microsp_and_host_names.apply(
        lambda x: predict_microsp_and_hosts(x.title_abstract),
        axis=1,
        result_type='expand'
    )

    # assign new column extracting verbs from sentences in title_abstract,
    # where sentences have both microsporidia + host mentions
    #
    # use this to help distinguish between microsporidia and host species more
    # precisely later
    microsp_and_host_names['verbs'] = microsp_and_host_names.apply(
        lambda x: get_verbs_in_microsp_and_host_sentences(x.title_abstract,
                                                          x.microsp_in_text,
                                                          x.hosts_in_text),
        axis=1
    )

    # create a dataframe of all the verbs that occur in microsporidia and host
    # sentences, and how often these verbs have occurred in texts
    verb_freqs = get_verb_frequencies(microsp_and_host_names.verbs)

    # create wordcloud of all verbs in microsporidia and host sentences
    make_verb_wordcloud(microsp_and_host_names)

    # make matcher for finding sentences with infection "trigger words",
    # for microsporidia-host relation extraction
    trigger_words_matcher = make_trigger_words_matcher(verb_freqs)

    # add column for predicted microsporidia - host relations in each text
    microsp_and_host_names['relations'] = microsp_and_host_names.apply(
        lambda x: get_relations_str(
            predict_microsp_host_relations(x.title_abstract,
                                           x.pred_microsp,
                                           x.pred_hosts,
                                           trigger_words_matcher)
        ),
        axis=1
    )

    # save naive microsporidia + host predictions to csv for now, and evaluate
    # accuracy of predictions
    microsp_and_host_names.to_csv(
        '../../../results/naive_microsporidia_and_host_name_predictions.csv'
        )

    # save cached predictions for future reference
    with open('cached_microsp_host_preds.pickle', 'wb') as p:
        pickle.dump(PREDICTIONS_CACHE, p)

################################################################################

## Helper functions for predicting microsporidia + host entities in texts

def get_cached_document(txt: str) -> spacy.tokens.doc.Doc:
    if txt not in DOC_CACHE:
        DOC_CACHE[txt] = nlp(txt)
    
    return DOC_CACHE[txt]


def predict_microsp_and_hosts(txt: str) -> Tuple[str, str]:
    """From a given string, txt, predict the novel Microsporidia species and
    host species from the text.
    
    Arguments:
        txt: string to look for Microsporidia and host species names

    Return:
        Tuple of predicted Microsporidia names and host names, each as semi-colon
        separated strings.

        Empty string when there's no predictions for Microsporidia or hosts

    """
    # fetch cached taxonerd predictions for this text, if possible
    if txt in PREDICTIONS_CACHE:
        pred_taxons = PREDICTIONS_CACHE[txt]
    else:
        pred_taxons = format_taxonerd_predictions(taxonerd.find_in_text(txt))
        PREDICTIONS_CACHE[txt] = pred_taxons

    # predict microsporidia based on presence of known microsporidia genus
    # name in prediction
    microsp = [pred for pred in pred_taxons if re.search(MICROSP_GENUSES, pred[0])]
    microsp = microsp + \
        [pred for pred in pred_taxons if any([is_abbrev(pred[0], m[0]) for m in microsp])]
    
    # predict microsporidia based on taxons followed by new species indicators
    new_sp_spans = list(re.finditer(NEW_SPECIES_INDICATORS, txt))
    microsp = microsp + \
        [pred for pred in pred_taxons if any(
            [m.start() - pred[2] > 0 and m.start() - pred[2] < 15 for m in new_sp_spans]
        )]

    # all non-microsporidia predicted taxons are treated as hosts
    hosts = [pred for pred in pred_taxons if pred not in microsp]

    # return unique predicted microsporidia and host names, in that order
    return '; '.join(list(set([m[0] for m in microsp]))), \
        '; '.join(list(set(h[0] for h in hosts)))


def format_taxonerd_predictions(taxonerd_preds: pd.core.frame.DataFrame) -> \
    List[Tuple[str, int, int]]:
    formatted_preds = []
    for i in range(len(taxonerd_preds)):
        text = taxonerd_preds['text'][i]
        bounds = [int(j) for j in taxonerd_preds['offsets'][i].split()[1:]]
        formatted_preds.append((text, bounds[0], bounds[1]))

    # filter out obvious false positives
    # i.e 'microsporidia', 'microsporidium', or a family/order name (as indicated
    # # by a single token ending in 'ae')
    formatted_preds = [pred for pred in formatted_preds if pred[0] not in \
        {'Microsporidia', 'microsporidia', 'Microsporidium', 'microsporidium',
        'Microsporida', 'microsporida'} and not (len(pred[0].split()) == 1 and \
            re.search('ae$', pred[0]) is not None)]

    return formatted_preds


def is_abbrev(s1, s2):
    """Check if s1 is an abbreviated name to s2."""
    s2_abbrev = get_abbreviated_species_name(s2)
    if (s1 in s2_abbrev) or (s1 in s2_abbrev.replace(' ', '')) or \
        (s2_abbrev in s1) or (s2_abbrev.replace(' ', '') in s1):
        return True
    
    return False


def get_abbreviated_species_name(species: str) -> str:
    """Return the abbreviated version of a taxonomic name.
    If the taxonomic name is only 1 word, then return it as is.

    Ex: Culex pipiens -> C. pipiens
    Ex: Culex pipeins pipiens -> C. p. pipiens
    Ex: Culex -> Culex

    Arguments:
        species: the taxonomic name to abbreviate

    """
    if ' ' not in species:
        return species

    species = species.split()
    return f"{' '.join([s[0] + '.' for s in species[:len(species) - 1]])} {species[-1]}"

################################################################################

## Helper functions for finding common verbs (trigger words) in sentences
## indicating host infection by microsporidia


def get_verbs_in_microsp_and_host_sentences(txt: str, microsp: str, hosts: str) \
    -> Union[str, float]:
    """Return lemmas of all verbs in sentences containing both microsporidia and
    host mentions, from txt.
    
    Return NaN if no sentences containing both microsporidia and hosts, or
    no verbs all in such sentences
    """
    doc = get_cached_document(txt)

    try:
        microsp = re.split('; | \|\| ', microsp) if microsp != '' else []
        microsp = microsp + [get_abbreviated_species_name(m) for m in microsp]
    except (AttributeError, TypeError):
        microsp = []

    try:
        hosts = re.split('; | \|\| ', hosts) if hosts != '' else []
        hosts = hosts + [get_abbreviated_species_name(h) for h in hosts]
    except (AttributeError, TypeError):
        hosts = []

    sents = [sent for sent in doc.sents if \
        any([m in sent.text for m in microsp]) and any([h in sent.text for h in hosts])
    ]

    verbs = []
    for sent in sents:
        verbs.extend([tok.lemma_ for tok in sent if tok.pos_ == 'VERB'])

    return_str = '; '.join(list(set(verbs)))
    
    # Return NaN if no eligible verbs were found from txt
    return float('NaN') if return_str == '' else return_str


def get_verb_frequencies(verbs: pd.core.series.Series) -> pd.core.frame.DataFrame:
    """From a pandas series of verbs occurring in Microsporidia and host sentences
    from texts, return a dataframe with each verb and the frequencies in which they
    appear in texts.
    
    Arguments:
        verbs: Series of verb strings, taken from 'verbs' column of
        microsp_and_host_names

    Return:
        Dataframe with two columns: 'verb' and 'frequency'

    """
    verb_freqs = Counter(verbs.dropna().str.split('; ').explode())
    
    return pd.DataFrame(zip(verb_freqs.keys(), verb_freqs.values()),
        columns=['verb', 'freq']).sort_values(by=['freq'], ascending=False)


def make_verb_wordcloud(microsp_and_host_names: pd.core.frame.DataFrame) \
    -> None:
    """Make a word cloud of most commonly used verbs in sentences w/ microsporidia
    + host mentions in paper abstracts/titles.
    
    Save the word cloud the the results folder for this project.
    """
    # combine all verbs in the 'verbs' column into a string
    verbs = \
        ' '.join(list(np.concatenate(list(microsp_and_host_names[[
            'title_abstract', 'verbs'
            ]].drop_duplicates().dropna().verbs.str.split('; '))).flat))

    # create and save wordcloud
    wordcloud = WordCloud(width=1000, height=1000, background_color='white',
                          stopwords=STOPWORDS, min_font_size=10).generate(verbs)

    plt.figure(figsize = (10, 10), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    plt.savefig('../../../results/verbs_wordcloud.png')

################################################################################

## Helper functions for rule-based relation extraction between Microsporidia and
## host entities

def make_trigger_words_matcher(verb_freqs: pd.core.frame.DataFrame, n: int = 15) -> \
    spacy.matcher.matcher.Matcher:
    """Return a spaCy matcher 
    """
    trigger_words = list(filter(lambda v: v not in EXCLUDED_VERBS,
                                set(verb_freqs.verb[:n])))
    trigger_words_matcher = Matcher(nlp.vocab)
    trigger_words_matcher.add('trigger', [
        [{'LEMMA': {'IN': trigger_words}}]
    ])

    return trigger_words_matcher


def predict_microsp_host_relations(txt: str, pred_microsp: str, pred_hosts: str,
                                   trigger_words_matcher: spacy.matcher.matcher.Matcher) -> \
                                    Union[str, float]:
    """
    """
    if pred_microsp == '' or pred_hosts == '':
        # if no microsporidia or no host predictions for this text, then no
        # relations can be predicted
        return float('nan')

    doc = get_cached_document(txt)
    pred_microsp, pred_hosts = pred_microsp.split('; '), pred_hosts.split('; ')

    infection_sents = [sent for sent in doc.sents if trigger_words_matcher(sent)]
    microsp_host_relations = {}

    for sent in infection_sents:
        microsp_in_sent = {m for m in pred_microsp if m.lower() in sent.text.lower()}
        hosts_in_sent = {h for h in pred_hosts if h.lower() in sent.text.lower()}

        # microsporidia + host predictions both in the same sentence, so try
        # to extract relations from this sentence
        if microsp_in_sent and hosts_in_sent:
            # assign all hosts in sentence to each microsporidia
            for m in microsp_in_sent:
                if m not in microsp_host_relations:
                    microsp_host_relations[m] = hosts_in_sent
                
                else:
                    microsp_host_relations[m] = \
                        microsp_host_relations[m] | hosts_in_sent
    
    return microsp_host_relations if len(microsp_host_relations) > 0 else float('nan')


def get_relations_str(relations: Dict[str, Set[str]]) -> str:
    """
    """
    # Note to self: how to combine redundant entries in predictions
    # 1) get all dictionary keys, and determine which entries are synonyms
    # 2) merge together all synonymous entries, and keep abbreviated names
    # 3) determine which host entries are synonyms for each key
    # 4) remove synonymous names, keeping the full ones
    # 
    # when evaluating in R, check for both full names and abbreviated names
    # for match between predictions and recorded values
    if isinstance(relations, float):
        return float('nan')

    relations_copy = copy.deepcopy(relations)
    for m in relations.keys():
        abbrev_microsp = get_abbreviated_species_name(m)
        if m != abbrev_microsp and abbrev_microsp in relations:
            # merge entry for full microsporidia name with entry for abbreviated
            # microsporidia name, then remove the entry for the abbreviated name
            relations_copy[m] = relations_copy[m] | relations_copy[abbrev_microsp]
            relations_copy.pop(abbrev_microsp)

            # remove abbreviated names from predicted hosts for each microsporidia,
            # if no full name was also predicted for the host
            for h in relations[m] | relations[abbrev_microsp]:
                abbrev_host = get_abbreviated_species_name(h)
                if h != abbrev_host and abbrev_host in \
                    relations_copy[m]:
                    relations_copy[m].remove(abbrev_host)
    
    # return string of all microsporidia and host relations like so:
    # microsp 1: host 1, host 2, ...; microsp 2: host 1, host 2, ...; ...
    return \
        '; '.join([f"{m}: {', '.join(relations_copy[m])}" for m in relations_copy])

################################################################################

if __name__ == '__main__':
    main()