# -----------------------------------------------------------------------------
#
# Predict microsporidia polar tube coil measures from texts
#
# Jason Jiang - Created: Feb/07/2023
#               Last edited: Feb/08/2023
#
# Mideo Lab - Microsporidia text mining
#
#
# -----------------------------------------------------------------------------

import spacy
from spacy.matcher import Matcher
import numerizer  # for converting numeric tokens to numeric strings

import re
import pandas as pd
import numpy as np

from typing import List, Tuple

###############################################################################

nlp = spacy.load('en_core_web_sm')  # initialize spacy's small language model

EXAMPLES = [
    """A new species of a microsporidan, Abelspora portucalensis, was found in the hepatopancreas of Carcinus maenas. forming white xenomas. Each xenoma seems to consist of an aggregate of hypertrophic host cells in which the parasite develops and proliferates. This cytozoic microsporidan being characterized by one uninucleate schizont giving rise to two sporonts. each originating two sporoblasts, resulting in two spores within a persistent sporophorous vacuole (pansporoblast) should be included in a new family Abelsporidue. In fresh smears most spores were 3.1-3.2 um long and 1.2-1.4 um wide. Fixed, stained, and observed in SUS mature spores measured 3.1 ± 0.08 x 1.3 ± 0.06 um (n = 25 measurements). Spore cytoplasm was dense and granular. Polyribosomes were arranged in helicoidal tape form. The polar filament was anisofilar and consisted of a single coil with 5-6 turns. The anchoring disc and and the anterior zone of the filament are surrounded by the polaroplast composed of two usual zones. In the anterior zone, the membrane of the polar filament is in continuity with the membranes of the polaroplast. The appearance of a microsporidan with described nuclear divisions in life cycle. spores shape and size. polaroplast and polar filament morphology and identity of the host suggests that we may erect a new genus Abelspora and a new species A. portucalensis (Portugal = Portucalem).""",
    """A new microsporidian species was described from the hypoderm of Daphnia magna sampled from gibel carp (Carassius auratus gibelio) ponds located in Wuhan city, China. The infected cladocerans generally appeared opaque due to numerous plasmodia distributed in the host integument. The earliest stages observed were uninucleate meronts that were in direct contact with the host cell cytoplasm. Meronts developed into multinucleate sporogonial plasmodia enclosed in sporophorous vesicles. Sporoblasts were produced by the rosette-like division of sporogonial division. Mature spores were pyriform and monokaryotic, measuring 4.48 ± 0.09 (4.34-4.65) µm long and 2.40 ± 0.08 (2.18-2.54) µm wide. The polaroplast was bipartite with loose anterior lamellae and tight posterior lamellae. Polar filaments, arranged in two rows, were anisofilar with two wider anterior coils, and five narrower posterior coils. The exospore was covered with fibrous secretions and was composed of four layers. Phylogenetic analysis based on the obtained SSU rDNA sequence, indicated that the present species clustered with three unidentified Daphnia pulicaria-infecting microsporidia with high support values to form a monophyletic lineage, rather than with the congener, Agglomerata cladocera. The barcode motif of the internal transcribed spacer (ITS) region of the novel species was unique among representatives of the "Agglomeratidae" sensu clade (Vávra et al., 2018). Based on the morphological characters and SSU rDNA-inferred phylogenetic analyses, a new species was erected and named as Agglomerata daphniae n. sp. This is the first report of zooplankton-infecting microsporidia in China.""",
    """Agglomerata lacrima n. sp. is the first species of the genus described from a copepod host (Acanthocyclops vernalis.) It was studied using light- and electron-microscopy. All stages of the life-cycle had isolated nuclei. The earliest stages found were uninucleate merozoites. Sporogony produced 4-12 (mostly 8) pyriform spores by rosette-like budding within a fragile sporophorous vesicle. Live spores measured 4.4 +/- 0.2 x 2.6 +/- 0.2 μm, and fixed spores measured 3.7 +/- 0.2 x 1.6 +/- 0.2 μm. The exospore was constructed of 4 layers. The anisofilar polar filament made 5-6 coils in the posterior half of the spore. The polaroplast had an anterior part with wide lamellae (chambers), followed by a second zone of narrow lamellae. Tubule-like structures which might constitute a third polaroplast region were present immediately anterior to the first filament coil. Cytological characteristics and the generic position of the species are discussed, and it is compared to related or resembling species, and to all previously reported microsporidian species from copepods.""",
    """The new microsporidium Agglomerata volgensae n. sp. is described based on light microscopic and ultrastructural characteristics. The parasite invades the hypoderm and adipose tissue of Daphnia magna and causes hypertrophy. All life cycle stages have isolated nuclei. Sporogonial plasmodia divide by rosette-like division, producing 4-16, usually 8, sporoblasts. A sporophorous vesicle is initiated after the first nuclear division of the sporont. The fragile vesicle either collects all daughter cells of the sporont, or the vesicle divides together with the plasmodium to enclose spores in individual sporophorous vesicles. Fibril-like projections connect the exospore with the envelope of the sporophorous vesicle. Tubules, with walls of exospore material, are formed together with the sporoblasts. Mature spores are pyriform with pointed anterior pole and an obliquely positioned posterior vacuole. Unfixed spores measure 3.2-3.7x1.7-2.0 μm. The exospore is layered, approximately 38 nm thick. The polar filament is lightly anisofilar with 2-3 wide anterior coils, and 2-3 more narrow posterior coils, in a single layer of coils in the posterior half of the spore. The polaroplast has two regions: irregular wide lamellae or chambers surrounding the straight part of the filament, and more loosely arranged narrow lamellae in the coil region. The discrimination from other microsporidian species and the systematic position are briefly discussed.""",
    """On the basis of electronmicroscopic data the description of Alfvenia ceriodaphniae sp. n. from Ceriodaphnia reticulata (Crustacea, Cladocera) is presented. The nuclear apparatus of meronts is diplokariotic. Sporonts, sporoblasts and spores are uninuclear. The sporogonial Plasmodium undertakes the rosette-like division with the formation of sporoblasts. Spores are egg-shaped, their size: 4.5(4.2-4.8) X 3.3(2.9-3.5) ц т . Evry spore lays individually inside the envelope, that is formed from the external layer of the sporoblast wall. Polar tube is isofillar, forming 8 - 9 coils. Polaroplast is consisted of the lamellar and chamber parts. The site of parasite localisation is crustacean hypoderm."""
]

matcher = Matcher(nlp.vocab)

coil_pattern = [
    {'LIKE_NUM': True},
    {'TEXT': {'REGEX': ".+"}, 'OP': '{,3}'},
    {'LIKE_NUM': True, 'OP': '?'},
    {'TEXT': {'REGEX': ".+"}, 'OP': '{,3}'},
    {'LEMMA': {'REGEX': '(coil|spire|twist|turn)'}}
]

matcher.add('polar_coil', [coil_pattern])

###############################################################################

def main():
    pass

###############################################################################

## Helper functions

def predict_polar_tube_measures(text: str) -> List[np.ndarray]:
    doc = nlp(text)

    # extract coil measures from polar tube sentences, treating the sum of
    # all coil measures from each sentence as the coils for a distinct spore
    # class, or microsporidia species
    polar_coils = [
        get_sentence_coil_measure(sent) for sent in doc.sents if \
            re.search('polar +(tube|tubule|filament)', sent.text.lower()) and \
                re.search('(coil|twist|turn|spire)', sent.text.lower())
    ]

    # for each sentence, treat as one "coil measure"
    # accumulate list of such measures, with each measure represented by a
    # length 1 - 2 tuple
    return polar_coils


def get_sentence_coil_measure(sent: spacy.tokens.span.Span) -> np.ndarray:
    # get polar coil pattern matches within sentence
    matches = matcher(sent)

    # resolve overlapping matches, removing the shorter of overlapping matches
    # and only keeping the longest one
    resolved_matches = resolve_overlapping_matches(matches)

    # return matches for polar coil measures as a tuple of integers
    # length 1 tuple if a single value, length 2 if a range of coils is given
    return format_coil_matches_as_numbers(resolved_matches, sent)


def resolve_overlapping_matches(matches: List[Tuple[int]]) -> List[Tuple[int]]:
    # overlapping coil matches will have same span end within a sentence
    # so, take the longest coil match (which comes after the shorter span),
    # and set the shorter overlapping match to None
    for i in range(1, len(matches)):
        if matches[i - 1][2] == matches[i][2]:
            matches[i - 1] = None
    
    return [match for match in matches if match is not None]


def format_coil_matches_as_numbers(resolved_matches: List[Tuple[int]],
                                   sent: spacy.tokens.span.Span) -> np.ndarray:
    coil_measures = []

    for match in resolved_matches:
        sent_match = sent[match[1] : match[2]]

        coil_measure = \
            [float(tok._.numerized) for tok in sent_match if tok.pos_ == 'NUM']

        coil_measures.append(np.array(coil_measure))

    return sum(coil_measures)

###############################################################################

if __name__ == '__main__':
    main()