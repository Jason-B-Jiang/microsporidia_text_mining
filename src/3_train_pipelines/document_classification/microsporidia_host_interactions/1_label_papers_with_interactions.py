# -----------------------------------------------------------------------------
#
# Label paper abstracts with presence or absence of novel microsporidia-host
# interactions
#
# Jason Jiang - Created: 2022/11/10
#               Last edited: 2022/11/24
#
# Mideo Lab - Microsporidia text mining
#
# -----------------------------------------------------------------------------

import pandas as pd
import spacy
from typing import Union, List, Set, Dict, Tuple
from collections import Counter
from string import punctuation
import numpy as np
import pickle

###############################################################################

def main() -> None:
    nlp = spacy.load('en_core_web_sm')

    # dataset of paper abstracts and microsporidia/host entities found within
    interactions = pd.read_csv(
        '../../../../data/formatted_host_microsp_names.csv'
    )[['title_abstract', 'microsp_in_text_matches', 'hosts_in_text_matches']]

    # convert each paper abstract into a spaCy document, then simplify
    # each document into a list of non-stop words.
    #
    # this makes downstream analyses much simpler
    doc_words = convert_docs_to_word_lists(
        [nlp(txt) for txt in interactions['title_abstract']],
        nlp.Defaults.stop_words)

    # add column indicating whether text has novel interaction recorded
    # i.e: both microsporidia + host entity are found in text
    interactions['interaction'] = interactions.apply(
        lambda row: paper_has_recorded_interactions(row['microsp_in_text_matches'],
        row['hosts_in_text_matches']),
        axis=1
        )

    # get top n words appearing across texts
    top_n_words = get_top_n_words_from_docs(doc_words)

    # get inverse document frequency for each of the top n words
    top_n_words_idf = get_words_idf_across_docs(top_n_words, doc_words)

    # match document for each paper abstract to their interaction label, and
    # a vector for the term frequency-inverse document frequency for each
    # of the top n words in the abstract
    labelled_dataset = make_labelled_dataset(doc_words,
                                             interactions['interaction'],
                                             top_n_words,
                                             top_n_words_idf)
    
    # finally, write this dataset of labelled paper abstracts and tf-idfs
    # as a pickle object
    with open('labelled_interactions.pickle', 'wb') as f:
        pickle.dump(labelled_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

###############################################################################

## Helper functions

def convert_docs_to_word_lists(docs: List[spacy.tokens.doc.Doc],
                               stopwords: Set[str]) -> \
    List[Tuple[spacy.tokens.doc.Doc, List[str]]]:
    """
    For each doc in docs, convert the spaCy document into a list of acceptable
    words.

    I.e: not in a defined set of stop words, not punctuation and not a number.
    """
    word_lsts = []

    # note: I'm using both using string.punctuation and spaCy's part-of-speech
    # tagging to find and exclude punctuation tokens.
    #
    # I found that "%" was often missed as punctuation by spaCy, so I'm
    # using string.punctuation to account for this
    for doc in docs:
        word_lsts.append(
            (doc,
            [w.lemma_.lower() for w in doc if w.lemma_ not in stopwords and \
                w.lemma_ not in punctuation and w.pos_ != 'PUNCT' \
                    and w.pos_ != 'NUM'])
        )
    
    return word_lsts


def convert_matches_to_nan(matches: Union[str, float]) -> Union[str, float]:
    """
    Placeholder docstring.
    """
    if pd.isnull(matches):
        return matches
    
    # if matches are all NA, then we have no matches, so return NaN
    matches_split = list(set(matches.split(' || ')))
    if len(matches_split) == 1 and matches_split[0] == 'NA':
        return float('nan')

    return matches


def paper_has_recorded_interactions(microsp_in_text_matches: Union[str, float],
                                    hosts_in_text_matches: Union[str, float]) ->\
                                        bool:
    """
    Placeholder docstring.
    """
    microsp_in_text_matches = convert_matches_to_nan(microsp_in_text_matches)
    hosts_in_text_matches = convert_matches_to_nan(hosts_in_text_matches)

    if not isinstance(microsp_in_text_matches, float) and \
        not isinstance(hosts_in_text_matches, float):
        return True

    return False


def get_top_n_words_from_docs(doc_words: List[Tuple[spacy.tokens.doc.Doc, List[str]]],
                              n: int = 100) -> List[str]:
    """
    Get the top n words appearing in a list of spaCy documents.

    Exclude stop words + punctuation + numbers in getting these top n words,
    and use word lemmas when considering word frequencies.

    Default n is 100, unless otherwise specified.
    """
    words = []
    for doc in doc_words:
        words.extend(doc[1])

    words_freq = Counter(words)

    # return top n highest frequency words from docs
    return sorted(list(set(words)),
                  reverse=True,
                  key=lambda word: words_freq[word])[:n]


def get_words_idf_across_docs(words: List[str],
                              doc_words: Dict[spacy.tokens.doc.Doc, List[str]]) \
                                -> Dict[str, float]:
    """
    Return a dictionary for the term inverse document frequency (idf) for
    each word in words.

    idf gives a weight for how "important" a word is across documents.

    idf for a word is calculated as follows, where +1 is added to number of
    documents to prevent division by zero:
        ln(number of documents / number of documents containing word + 1)
    """
    # extract only values (i.e: lists of acceptable words from each document)
    doc_words_values = [doc[1] for doc in doc_words]
    
    # create a dictionary mapping each word in words to their inverse document
    # frequencies across documents in doc_words
    words_idf = {}
    for word in words:
        words_idf[word] = np.log(
            len(doc_words) / (len([d for d in doc_words_values if word in d]) + 1)
        )

    return words_idf


def get_tf_idf_vector(words: List[str], words_idf: Dict[str, float],
                      doc_words: List[str]) -> np.ndarray:
    """
    Return an array representing the term frequency-inverse document frequencies
    (tf-idf) for each word in words, for the document doc_words.

    tf-idf for a word is calculated as follows:
    # of times word appears in document * inverse document frequency of word
    """
    return np.asarray(
        [len([w for w in doc_words if w == word]) * words_idf[word] for \
            word in words]
    )


def make_labelled_dataset(doc_words: List[Tuple[spacy.tokens.doc.Doc, List[str]]],
                          labels: pd.core.series.Series,
                          words: List[str],
                          words_idf: Dict[str, float]) -> List[dict]:
    """
    Return a list of dictionaries, where each dictionary holds the spaCy
    document for a paper abstract, a label (T or F) of whether the abstract
    has any novel microsporidia-host interactions, and a vector storing the
    term frequency–inverse document frequencies for the abstract.

    Precondition: len(doc_words) == len(labels)
    """
    return [{'doc': doc_words[i][0],
             'label': labels[i],
             'vector': get_tf_idf_vector(words, words_idf, doc_words[i][1])} for \
                i in range(len(doc_words))]

###############################################################################

if __name__ == '__main__':
    main()