
# -----------------------------------------------------------------------------
#
# Predict microsporidia polar tube coil measures from texts
#
# Jason Jiang - Created: Feb/07/2023
#               Last edited: Mar/02/2023
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
    """On the basis of electronmicroscopic data the description of Alfvenia ceriodaphniae sp. n. from Ceriodaphnia reticulata (Crustacea, Cladocera) is presented. The nuclear apparatus of meronts is diplokariotic. Sporonts, sporoblasts and spores are uninuclear. The sporogonial Plasmodium undertakes the rosette-like division with the formation of sporoblasts. Spores are egg-shaped, their size: 4.5(4.2-4.8) X 3.3(2.9-3.5) ц т . Evry spore lays individually inside the envelope, that is formed from the external layer of the sporoblast wall. Polar tube is isofillar, forming 8 - 9 coils. Polaroplast is consisted of the lamellar and chamber parts. The site of parasite localisation is crustacean hypoderm.""",
    # weird examples start here
    """A new microsporidian parasite of a freshwater cladoceran from southern Sweden is described using light and electron microscopical methods. Development comprises 2 merogonial sequences, the first resulting in a cluster of 8 merozoites, the second in a chain of 4 merozoites. Each secondary merozoite develops into a sporont which divides into 2 sporoblasts, each of which develops into a spore. The spores are broadly oval and in fresh smears measure about 6 fim in length, with a single nucleus and a posterosome. The polar filament is about 40 fim long, of even thickness throughout, and appears as 15—18 coils in a single layer. The anchoring disc is small and the polaroplast is composed of 2 lamellar parts. Outside the plasma membrane of the sporont a 5-layered, electrondense substance is produced, which further differentiates into endo- and exospore, an electron-dense substance occurring patchily on the exospore and a pansporoblast membrane. During development the sporoblasts and the young spores are connected by a dense substance. Mature spores appear single or paired. The pansporoblast membrane is composed of 2 structurally different layers, namely a thin outer, single membrane and an inner layer composed of tubular structures. It is connected to the spore coat by patches of the dense substance. The new microsporidium is considered to belong to a new genus of the family Telomyxidae, and its systematic relationship with this and the related family Tuzetiidae is discussed. A survey of microsporidia from Cladocera is included.""",
    """Canningia spinidentis gen. Et sp. n. infects the fir bark beetle Pityokteines spinidens Rtt. In Austria. The pathogen attacks mainly the fat body, Malpighian tubules, the muscles and the connective tissue of larvae and adults, and the gonads of adults. The development is haplokaryotic, with single spores. Spores are short tubular, uninucleate with globular anchoring disc inserted subapically, laterally, in a depression of the endospore wall. Polar filament is isofilar with 5/6 coils. Polaroplast is composed of two lamellar parts of different density. A new genus Canningia gen. n. is proposed based on differences in ultrastructures of spores from Unikaryon Canning, Barker, Hammond et Nicholas, 1974.""",
]

matcher = Matcher(nlp.vocab)

coil_pattern = [
    {'POS': 'NUM'},
    {'TEXT': {'REGEX': "^(?!.m$).*"}, 'OP': '{,2}'},
    {'POS': 'NUM', 'OP': '?'},
    {'TEXT': {'REGEX': "^(?!.m$).*"}, 'OP': '{,2}'},
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
                matcher(sent)
    ]

    # for each sentence, treat as one "coil measure"
    # accumulate list of such measures, with each measure represented by a
    # length 1 - 2 tuple
    return polar_coils


def get_sentence_coil_measure(sent: spacy.tokens.span.Span) -> np.ndarray:
    # corner case: uncoiled polar tube = 0 coils
    if re.search("(uncoiled|noncoiled|non-coiled)", sent.text.lower()):
        return np.array([0.0])

    # get polar coil pattern matches within sentence
    matches = matcher(sent)

    # resolve overlapping matches, removing the shorter of overlapping matches
    # and only keeping the longest one
    resolved_matches = resolve_overlapping_matches(matches)

    # return matches for polar coil measures as an array of floats
    # length 1 array if a single value, length 2 if a range of coils is given
    return format_coil_matches_as_numbers(resolved_matches, sent)


def resolve_overlapping_matches(matches: List[Tuple[int]]) -> List[Tuple[int]]:
    # overlapping coil matches will have same span end within a sentence
    # so, take the longest coil match (which comes after the shorter span),
    # and set the shorter overlapping match to None
    for i in range(1, len(matches)):
        if matches[i - 1][2] == matches[i][2]:
            matches[i - 1] = None

    first_resolved = [m for m in matches if m is not None]
    for i in range(1, len(first_resolved)):
        if first_resolved[i - 1][1] > first_resolved[i][0]:
            first_resolved[i] = (first_resolved[i - 1][0], first_resolved[i][1])
            first_resolved[i - 1] = None
    
    return [match for match in first_resolved if match is not None]


def format_coil_matches_as_numbers(resolved_matches: List[Tuple[int]],
                                   sent: spacy.tokens.span.Span) -> np.ndarray:
    coil_measures = []

    for match in resolved_matches:
        coil_measure = convert_coil_measure_to_numeric(sent[match[1] : match[2]])

        # a half-coil was indicated as a fraction, so add the fractional part
        # to the whole coil count
        if len(coil_measure) == 2 and coil_measure[1] < 1.0:
            coil_measure = np.array([coil_measure[0] + coil_measure[1]])

        # a range of coils for polar tube should only be two numbers max
        # i.e: 1 coil, 2 - 3 coils
        # so if we get more than three numbers for a range, something weird
        # was extracted, so don't add that to our coil measures
        if len(coil_measure) <= 2:
            coil_measures.append(coil_measure)

    return sum(coil_measures)


def convert_coil_measure_to_numeric(match: spacy.tokens.span.Span) -> np.ndarray:
    numeric_coil_measures = []

    for measure in [tok for tok in match if tok.pos_ == 'NUM']:
        try:
            numeric_coil_measure = np.array([float(measure._.numerized)])

        except ValueError:
            
            if '/' in measure.text:
                 # fraction was not properly numerized
                numeric_coil_measure = \
                    np.array([float(measure.text.split("/")[0]) / float(measure.text.split("/")[1])])

            elif re.search('\\d+.\\d+', measure.text):
                # some kind of weirdly delimited coil range
                split_ = re.split('[^0-9\\.]', measure.text)
                numeric_coil_measure = \
                    np.sort(np.array([float(numerizer.numerize(split_[0])), float(numerizer.numerize(split_[1]))]))
            
            else:
                numeric_coil_measure = None
        
        if numeric_coil_measure is not None:
            numeric_coil_measures.append(numeric_coil_measure)
    
    return sum(numeric_coil_measures)
    # return np.sort(np.array(numeric_coil_measures).flatten())

###############################################################################

if __name__ == '__main__':
    main()

# TODO - troubleshoot funny outputs

# pt_df = pd.read_csv('../../../data/polar_coil_data/polar_coils.csv')

# preds = []
# errors = []

# for text in pt_df['abstract']:
#     try:
#         preds.append((text, predict_polar_tube_measures(text)))

#     except ValueError:
#         errors.append(text)

triple_array = 'Collections of the dicyemid mesozoan Kantharella antarctica were made in the Weddell Sea during the Antarctic Expedition of the research vessel B.V Polarstern in 1990 and 1991. A diplokaryotic microsporidian was found infecting all nematogens from all the samples taken in both years. The infected cells contained all developmental stages. Merogony initially was monokaryotic and sporogony of diplokaryotic sporonts was by multiple fission. The stained ovoidal spores measured between 4.3-6 μm x 1.7-2.3 μm. The ultrastructural findings come from 11 specimens of Kantharella antarctica that were cut in serial sections. All developmental stages were noteworthy because of the myelinosomes situated adjacent to each diplokaryon. Similarly conspicuous were some organelles in the spore: a prominent, extraordinarily electron dense anterior portion of the polaroplast and the posterior vacuole. The isofilar polar filament with a diameter of about i 15 nm showed 9-11 coils. The great number of empty spore cases together with an extruded polar filament are indicative of an autoinfection. Though these characteristics resemble in part those of the genus Nosema from the family Nosematidae, the species in Kantharella antarctica differs from the former by its unusual development, life cycle and unusual host. Thus, this new species has been placed in a new genus and the name Wittmannia antarctica proposed.'
weird_zero = 'A microsporidium was found infecting the fat body of larvae and adults of both sexes of Culex pipiens in Egypt. Developmental stages were found in larvae but only masses of spores were present in adults. The infection was easily visible in live mosquito larvae, as one or two blocks of opaque whitish fat body visible through the cuticle in each segment. Meronts were rounded cells, which were bounded by an unthickened unit membrane and divided by binary fission (rarely into four). At the onset of sporogony the surface membrane was thickened by electron dense deposits. This coat was sloughed off to form the sporophorous vesicle, the separation from the sporont surface being effected by the secretion of metabolic products into the sporophorous vesicle cavity. Division within the vesicle gave rise to eight uninucleate sporoblasts, then uninucleate spores. Spores exhibited an exospore of two membrane-like layers and a subtending layer of moderate electron density, appearing as eight to ten strata separated by fine lines and permeated by amorphous material, and an electron lucent endospore. The polar tube was anisofilar with 3–4 broad coils and 4–3 narrow coils. The development and spore structure were in accord with the genus Amblyospora Hazard and Oldacre, 1975 and, on the basis of spore size and number of coils of the polar tube, it is considered to be a new species, Amblyospora egypti n.sp.'
multiple_error_1 = 'Polar filaments, arranged in two rows, were anisofilar with two wider anterior coils, and five narrower posterior coils'
multiple_error_2 = 'The polar filament is lightly anisofilar with 2-3 wide anterior coils, and 2-3 more narrow posterior coils, in a single layer of coils in the posterior half of the spore.'

error_1 = "The polar tube was arranged in one row of 13–18 coils including 0–3 distal coils of lesser diameter."

predict_polar_tube_measures(error_1)