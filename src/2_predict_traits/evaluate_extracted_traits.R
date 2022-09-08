# -----------------------------------------------------------------------------
#
# Evaluate accuracy of predicted microsporidia traits
#
# Jason Jiang - Created: 2022/05/17
#               Last edited: 2022/08/26
#
# Mideo Lab - Microsporidia text mining
#
#
# -----------------------------------------------------------------------------

library(tidyverse)

################################################################################

## Evaluate polar tube coil predictions

WORDS_TO_DIGITS <- list('one' = 1, 'two' = 2, 'three' = 3, 'four' = 4,
                        'five' = 5, 'six' = 6, 'seven' = 7, 'eight' = 8,
                        'nine' = 9, 'ten' = 10)

PT_COIL_RANGE <- '\\d{1,2}\\.?\\d?( to | or | and | ?- ?| ?– ?| ?— ?|\\/)?\\d*\\.?\\d?'

date_to_range <- function(pt_coils_range) {
  # ---------------------------------------------------------------------------
  # Turn ranges that Excel turned into dates, back into ranges
  # ---------------------------------------------------------------------------
  if (!str_detect(pt_coils_range, '2022-')) {
    return(pt_coils_range)
  }
  
  pt_coils_range <- str_c(
    as.integer(str_split(str_remove(pt_coils_range, '2022-'), '-')[[1]]),
    collapse = '-'
  )
  
  return(pt_coils_range)
}

turn_number_words_to_digits <- function(pt_coil_pred) {
  # ---------------------------------------------------------------------------
  # Replace number words in extracted polar tube coil data with digits
  # ex: 'five to six coils' -> '5 to 6 coils'
  # ---------------------------------------------------------------------------
  # get individual coil predictions
  pt_coil_pred <- str_split(pt_coil_pred, ' \\|\\|\\| ')[[1]]
  
  for (i in 1 : length(pt_coil_pred)) {
    pred <- str_split(pt_coil_pred[i], ' ')[[1]]  # split into words
    
    for (j in 1 : length(pred)) {
      if (tolower(pred[j]) %in% names(WORDS_TO_DIGITS)) {
        pred[j] <- WORDS_TO_DIGITS[[tolower(pred[j])]]
      }
    }
    
    pt_coil_pred <- replace(pt_coil_pred, i, str_c(pred, collapse = ' '))
  }
  
  return(str_c(pt_coil_pred, collapse = ' ||| '))
}


extract_pt_coil_range <- function(pt_coil_pred) {
  return(
    str_c(sapply(str_split(pt_coil_pred, ' \\|\\|\\| ')[[1]],
                         function(s) {str_extract(s, PT_COIL_RANGE)}),
                        collapse = ' ||| ')
  )
}


convert_pt_coil_ranges_to_medians <- function(pred_pt_coil_formatted) {
  pt_coil_preds <- str_split(pred_pt_coil_formatted, ' \\|\\|\\| ')[[1]]
  
  for (i in 1 : length(pt_coil_preds)) {
    # get average/median of a range of polar tube coils
    pt_coil_preds[i] <-
      mean(
        as.numeric(str_split(pt_coil_preds[i], '( to | or | and | ?- ?| ?– ?| ?— ?|\\/)')[[1]])
        )
  }
  
  return(str_c(pt_coil_preds, collapse = '; '))
}


get_pt_coil_precision <- function(pt_coils_avg, pred_pt_coil_avg) {
  pt_coils_avg <- str_split(pt_coils_avg, '; ')[[1]]
  pred_pt_coil_avg <- str_split(pred_pt_coil_avg, '; ')[[1]]
  
  true_pos <- length(pred_pt_coil_avg[pred_pt_coil_avg %in% pt_coils_avg])
  false_pos <- length(pred_pt_coil_avg[!(pred_pt_coil_avg %in% pt_coils_avg)])
  
  return(true_pos / (true_pos + false_pos))
}


get_pt_coil_recall <- function(pt_coils_avg, pred_pt_coil_avg) {
  pt_coils_avg <- str_split(pt_coils_avg, '; ')[[1]]
  pred_pt_coil_avg <- str_split(pred_pt_coil_avg, '; ')[[1]]
  
  true_pos <- length(pred_pt_coil_avg[pred_pt_coil_avg %in% pt_coils_avg])
  false_neg <- length(pt_coils_avg[!(pt_coils_avg %in% pred_pt_coil_avg)])
  
  return(true_pos / (true_pos + false_neg))
}


pt_coil_preds <- read_csv('../../results/microsp_pt_predictions.csv') %>%
  rowwise() %>%
  mutate(pt_coils_range = ifelse(!is.na(pt_coils_range),
                                        date_to_range(pt_coils_range),
                                        NA),
         pred_pt_coil_formatted = extract_pt_coil_range(turn_number_words_to_digits(pred_pt_coil)),
         pred_pt_coil_avg = convert_pt_coil_ranges_to_medians(pred_pt_coil_formatted)) %>%
  filter(!is.na(pred_pt_coil) | !is.na(pt_coils_range) | !is.na(pt_coils_avg)) %>%
  mutate(precision = get_pt_coil_precision(pt_coils_avg, pred_pt_coil_avg),
         recall = get_pt_coil_recall(pt_coils_avg, pred_pt_coil_avg))
  

################################################################################

## Evaluate polar tube length predictions

extract_pt_length_value <- Vectorize(function(pred_pt) {
  # ---------------------------------------------------------------------------
  # Extract polar tube length from string where length was found, and convert
  # ranges of length to averages.
  # ---------------------------------------------------------------------------
  mean(as.numeric(
    str_split(str_extract(pred_pt, '\\d+\\.?\\d?(–|-| to )?\\d*\\.?\\d?'),
            '(–|-| to )')[[1]]
  ))
})


format_recorded_pt_length <- Vectorize(function(pt_avg) {
  as.numeric(str_split(pt_avg, ' ')[[1]][1])
})

pt_len_preds <- read_csv('../../results/microsp_pt_predictions.csv') %>%
  select(species, title_abstract, pred_pt, pt_max, pt_min, pt_avg) %>%
  filter(!is.na(pred_pt) | !is.na(pt_max) | !is.na(pt_min) | !is.na(pt_avg)) %>%
  mutate(pred_pt_formatted = ifelse(is.na(pred_pt), NA, extract_pt_length_value(pred_pt)),
         pt_avg_formatted = format_recorded_pt_length(pt_avg),
         tp = ifelse(!is.na(pred_pt_formatted),
                     as.numeric(pred_pt_formatted == pt_avg_formatted),
                     0),
         fp = as.numeric(!is.na(pred_pt_formatted) & pred_pt_formatted != pt_avg_formatted),
         fn = as.numeric(is.na(pred_pt_formatted) & !is.na(pt_avg_formatted)))

pt_len_precision <- sum(pt_len_preds$tp) / (sum(pt_len_preds$tp) + sum(pt_len_preds$fp))
pt_len_recall <- sum(pt_len_preds$tp) / (sum(pt_len_preds$tp) + sum(pt_len_preds$fn))

################################################################################

## Evaluate microsporidia species names + hosts predictions (naive predictions)

get_species_matches_in_text <- function(species, text) {
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  # get abbreviations for species names and join together w/ full names
  # get indices of species names occurrences in text
  if (is.na(species)) {
    return(NA)
  }
  
  # only consider unique lowercase species matches, to remove redundant predictions
  # that differ only by capitalization
  #
  # ex: Aedes punctor and aedes punctor are both predicted from the text, but will
  # be treated as one in the same
  species <- unique(tolower(str_split(species, '; ')[[1]]))
  
  matches <- str_locate_all(tolower(text), fixed(species))
  matches <- matches[lapply(matches, length) > 0]
  
  return (
    ifelse(length(matches) > 0,
           str_c(
             as.character(lapply(matches, function(x) {str_c(x[1], x[2], sep = '-')})),
             collapse = '; '),
           NA)
  )
}


remove_na_from_species <- Vectorize(function(species) {
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  species <- str_split(species, '; ')[[1]]
  to_return <- str_c(species[species != 'NA'], collapse = '; ')
  
  return(ifelse(to_return != '', to_return, NA))
})


get_tp_species <- Vectorize(function(pred_matches, actual_matches, strict) {
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  if (is.na(actual_matches)) {
    return(0)
  }
  
  if (!is.na(pred_matches)) {
    pred_matches <- str_split(pred_matches, '; ')[[1]]
  } else {
    pred_matches <- character(0)
  }
  
  actual_matches <- str_split(actual_matches, '; ')[[1]]
  
  if (strict) {
    return(length(actual_matches[actual_matches %in% pred_matches]))
  }
  
  
}, vectorize.args = c('pred_matches', 'actual_matches', 'strict'))


convert_span_to_range <- Vectorize(function(span) {
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  span <- str_split(span, '-')[[1]]
  return(span[1] : span[2])
})


get_fp_species <- Vectorize(function(preds, tp) {
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  if (is.na(preds)) {
    return(0)
  }
  
  return(length(str_split(preds, '; ')[[1]]) - tp)
}, vectorize.args = c('preds', 'tp'))


get_fn_species <- Vectorize(function(actual, tp) {
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  if (is.na(actual)) {
    return(0)
  }
  
  return(length(str_split(actual, '; ')[[1]]) - tp)
}, vectorize.args = c('actual', 'tp'))

# Load in same test data split used for evaluating spaCy models
test_species <-
  read_lines('../3_train_pipelines/microsp_host_relation_extraction/test_species.txt')

naive_name_preds <- read_csv('../../results/naive_microsporidia_and_host_name_predictions.csv',
                             show_col_types = F) %>%
  filter(species %in% test_species) %>%
  select(species, title_abstract, microsp_in_text, hosts_in_text,
         microsp_in_text_matches, hosts_in_text_matches, pred_microsp, pred_hosts) %>%
  # remove rows where none of the recorded hosts or microsporidia species are
  # in the title/abstract
  #
  # this info was only in the full text, so remove these entries for a more
  # fair comparison with the abstracts only
  filter(!is.na(hosts_in_text) | !is.na(microsp_in_text)) %>%
  # TODO - MOVE THIS CODE TO A HELPER R SCRIPT TO HOST/MICROSPORIDIA PREDICTION
  # CODE FOLDER, SO THE PREDICTED OUTPUTS WILL BE FORMATTED EXACTLY LIKE THIS
  group_by(title_abstract) %>%
  mutate(microsp_in_text = remove_na_from_species(
    str_c(str_replace_na(microsp_in_text), collapse = '; ')),
    hosts_in_text = remove_na_from_species(
      str_c(str_replace_na(hosts_in_text), collapse = '; ')),
    microsp_in_text_matches = remove_na_from_species(
      str_c(str_replace_na(microsp_in_text_matches), collapse = '; ')),
    hosts_in_text_matches = remove_na_from_species(
      str_c(str_replace_na(hosts_in_text_matches), collapse = '; '))) %>%
  distinct(title_abstract, .keep_all = T) %>%
  rowwise() %>%
  # add in columns for character span matches of predicted microsporidia and
  # hosts in texts
  mutate(pred_microsp_matches = get_species_matches_in_text(pred_microsp,
                                                            title_abstract),
         pred_hosts_matches = get_species_matches_in_text(pred_hosts,
                                                          title_abstract)) %>%
  ungroup() %>%
  # add in columns for true positives, false positives and false negative
  # predicted microsporidia/hosts, for calculating precision + recall
  #
  # Note: try 2 "levels" of true positive calculation:
  # 1) boundaries of prediction MUST match boundaries of actual entity (strict)
  # 2) boundaries of prediction CONTAINS the boundaries of the actual entity
  #    (permissive)
  mutate(tp_strict_hosts = get_tp_species(pred_hosts_matches, hosts_in_text_matches,
                                          strict=TRUE),
         tp_permissive_hosts = get_tp_species(pred_hosts_matches, hosts_in_text_matches,
                                              strict=FALSE),
         fp_strict_hosts = get_fp_species(pred_hosts_matches, tp_strict_hosts),
         fp_permissive_hosts = get_fp_species(pred_hosts_matches, tp_permissive_hosts),
         fn_strict_hosts = get_fn_species(hosts_in_text_matches, tp_strict_hosts),
         fn_permissive_hosts = get_fn_species(hosts_in_text_matches, tp_permissive_hosts),
         tp_strict_microsp = get_tp_species(pred_microsp_matches, microsp_in_text_matches,
                                            strict=TRUE),
         tp_permissive_microsp = get_tp_species(pred_microsp_matches, microsp_in_text_matches,
                                                strict=FALSE),
         fp_strict_microsp = get_fp_species(pred_microsp_matches, tp_strict_microsp),
         fp_permissive_microsp = get_fp_species(pred_microsp_matches, tp_permissive_microsp),
         fn_strict_microsp = get_fn_species(microsp_in_text_matches, tp_strict_microsp),
         fn_permissive_microsp = get_fn_species(microsp_in_text_matches, tp_permissive_microsp))


# precision/recall calculations for hosts and microsporidia, using both
# strict and permissive matches for true positives
#
# Microsporidia strict precision: 46.5%
# Microsporidia strict recall: 74.9%
# Microsporidia strict F1: 57.4%
#
# Microsporidia permissive precision:
# Microsporidia permissive recall:
# Microsporidia permissive F1:
#
# Host strict precision: 15.3%
# Host strict recall: 77.1%
# Host strict F1: 25.6%
#
# Host permissive precision:
# Host permissive recall:
# Host permissive F1:
microsp_precision_strict <- 
  sum(naive_name_preds$tp_strict_microsp) / (sum(naive_name_preds$tp_strict_microsp) + sum(naive_name_preds$fp_strict_microsp))

microsp_recall_strict <- 
  sum(naive_name_preds$tp_strict_microsp) / (sum(naive_name_preds$tp_strict_microsp) + sum(naive_name_preds$fn_strict_microsp))

microsp_f1_strict <- 2 * ((microsp_precision_strict * microsp_recall_strict) / (microsp_precision_strict + microsp_recall_strict))

hosts_precision_strict <- 
  sum(naive_name_preds$tp_strict_hosts) / (sum(naive_name_preds$tp_strict_hosts) + sum(naive_name_preds$fp_strict_hosts))

hosts_recall_strict <- 
  sum(naive_name_preds$tp_strict_hosts) / (sum(naive_name_preds$tp_strict_hosts) + sum(naive_name_preds$fn_strict_hosts))

hosts_f1_strict <- 2 * ((hosts_precision_strict * hosts_recall_strict) / (hosts_precision_strict + hosts_recall_strict))

################################################################################

## Evaluate microsporidia + host name predictions (using verb information)

## TBA

################################################################################

## Evaluate Microsporidia infection site predictions

get_infection_tp_fp_fn <- function(recorded, predicted) {
  # ---------------------------------------------------------------------------
  # Docstring goes here
  # ---------------------------------------------------------------------------
  if (is.na(recorded)) {
    recorded <- character()
  } else {
    recorded <- str_split(recorded, '; ')[[1]]
  }
  
  if (is.na(predicted)) {
    predicted <- character()
  } else {
    predicted <- str_split(predicted, '; ')[[1]]
  }
  
  tp <- length(intersect(recorded, predicted))
  fp <- length(predicted) - tp
  fn <- length(recorded) - tp
  
  return(str_c(tp, fp, fn, sep = ','))
}

all_recorded_sites_in_text <- function(recorded, text) {
  recorded <- tolower(str_split(recorded, '; ')[[1]])
  text <- tolower(text)
  
  return(all(str_detect(text, recorded)))
}

microsp_infection_preds <- read_csv('../../results/microsp_infection_site_predictions.csv') %>%
  filter(num_papers < 2) %>%  # look at species with only 1 paper for now
  rowwise() %>%
  # for now, focus on cases where we're confident all recorded sites are in
  # abstract/title
  filter(all_recorded_sites_in_text(infection_site, abstract)) %>%
  mutate(tp_fp_fn = get_infection_tp_fp_fn(infection_site_normalized,
                                           pred_infection_site))  %>%
  separate(col= tp_fp_fn, into = c('tp', 'fp', 'fn'), sep = ',') %>%
  mutate(tp = as.integer(tp),
         fp = as.integer(fp),
         fn = as.integer(fn))

# 28% precision, 36% recall
# NEW: 29.6% precision, 37.8% recall
# 40% precision, 50% recall if exclude cases where tissues not in abstract
# NEW: 54.7% precision, 70.9% recall
infection_precision <-
  sum(microsp_infection_preds$tp) / (sum(microsp_infection_preds$tp) + sum(microsp_infection_preds$fp))

infection_recall <-
  sum(microsp_infection_preds$tp) / (sum(microsp_infection_preds$tp) + sum(microsp_infection_preds$fn))

################################################################################

## Evaluate predicted localities

get_regions_and_subregions_list <- function(locs) {
  locs <- str_split(locs, '; ')[[1]]
  regions <- sapply(locs, function(x) {str_split(x, ' ?\\(')[[1]][1]})
  subregions <-
    lapply(locs, function(x) {str_split(str_extract(x, '(?<=\\().+(?=\\))'), ' \\| ')[[1]]})
  
  names(subregions) <- regions
  
  return(subregions)
}


get_true_hits <- function(pred, recorded) {
  true_hits <- list()
  # for (p in pred)
  #   if p is a substring of anything in recorded, or if anything in recorded
  #   is a substring of p, append c(pred, recorded) to some accumulated vector
  #   and remove the match in recorded
  for (p in pred) {
    # predicted site is a substring of recorded site, or any recorded site is
    # a substring of the predicted site
    hits <- which(str_detect(recorded, p) | str_detect(p, recorded))
    
    if (length(hits) > 0) {
      for (h in hits) {
        true_hits[[length(true_hits) + 1]] <- c(p, recorded[h])
      }
      
      # remove matches so every recorded location may only match with one
      # predicted location at most
      recorded <- recorded[-hits]
    }
  }
  
  return(true_hits)
}


get_loc_tp <- function(pred, recorded) {
  pred <- get_regions_and_subregions_list(pred)
  recorded <- get_regions_and_subregions_list(recorded)
  
  # return list of vectors of matches
  # (pred_match, recorded_match, ...)
  intersecting_regions <- get_true_hits(names(pred), names(recorded))
  
  num_subregion_matches <- 0
  for (region in intersecting_regions) {
    intersecting_subregions <- get_true_hits(
      pred[region[1]], recorded[region[2]]
    )
    
    num_subregion_matches <- num_subregion_matches + length(intersecting_subregions)
  }
  
  return(length(intersecting_regions) + num_subregion_matches)
}

check_if_regions_in_abstract <- function(title_abstract, locality) {
  if (is.na(locality)) {
    return(FALSE)
  }
  
  locality <- str_split(locality, '; ')[[1]]
  regions <- sapply(locality, function(x) {str_extract(x, '.+(?= \\()')[[1]]})
  
  return(any(str_detect(title_abstract, regions)))
}


microsp_locality_preds <- read_csv('../../results/microsp_locality_predictions.csv') %>%
  # manual fix for correctly associating Siberia with Russia
  mutate(locality_normalized = ifelse(locality_normalized == 'Siberia ()',
                                      'Russia (Siberia)',
                                      locality_normalized),
         locality_normalized = ifelse(locality_normalized == 'Western Siberia (Novosibirsk)',
                                      'Russia (Novosibirsk | Siberia)',
                                      locality_normalized),
         locality_normalized = ifelse(locality_normalized == '(pond (Tom River, Tomsk, Western Siberia)',
                                      'Russia (Tom River | Tomsk | Siberia)',
                                      locality_normalized),
         locality_normalized = ifelse(locality_normalized == '(pond (Krotovo Lake, Troitskoe, Novosibirsk, Western Siberia)',
                                      'Russia (Krotovo Lake | Troitskoe | Novosibirsk region | Siberia)',
                                      locality_normalized)) %>%
  rowwise() %>%
  mutate(tp = get_locality_tp(pred_locality, locality_normalized),
         fp = get_locality_fp(pred_locality, locality_normalized),
         fn = get_locality_fn(pred_locality, locality_normalized)) %>%
  mutate(regions_in_abstract = check_if_regions_in_abstract(title_abstract, locality_normalized))

# 18% precision, 45% recall
# if excluding cases where regions not in abstract, 24% precision and 77% recall
locality_precision <-
  sum(microsp_locality_preds$tp) / (sum(microsp_locality_preds$tp) + sum(microsp_locality_preds$fp))

locality_recall <-
  sum(microsp_locality_preds$tp) / (sum(microsp_locality_preds$tp) + sum(microsp_locality_preds$fn))

################################################################################

## Evaluate predicted spore nuclei counts

get_nucleus_tp <- function(pred_nucleus, nucleus) {
  # ---------------------------------------------------------------------------
  # Docstring goes here
  # ---------------------------------------------------------------------------
  tp <- 0
  pred_nucleus <- sort(pred_nucleus)
  nucleus <- sort(nucleus)
  
  while (length(pred_nucleus) > 0 & length(nucleus) > 0) {
    if (pred_nucleus[length(pred_nucleus)] == nucleus[length(nucleus)]) {
      # if last elements of both vectors are equal, add 1 to true positive count
      # and remove last elements of both vectors
      tp <- tp + 1
      pred_nucleus <- pred_nucleus[-length(pred_nucleus)]
      nucleus <- nucleus[-length(nucleus)]
      
    } else if (pred_nucleus[length(pred_nucleus)] > nucleus[length(nucleus)]) {
      # last element of first vector is greater than last element of second
      # vector, remove last element of first vector
      pred_nucleus <- pred_nucleus[-length(pred_nucleus)]
      
    } else {
      # last element of second vector is greater than last element of first
      # vector, remove last element of second vector
      nucleus <- nucleus[-length(nucleus)]
    }
  }
  
  return(tp)
}

get_nucleus_tp_fp_fn <- function(pred_nucleus, nucleus) {
  # ---------------------------------------------------------------------------
  # Return comma separated string of number of true positive, false positive,
  # false negative spore nucleus predictions.
  #
  # Inputs:
  #   pred_nucleus: predicted number of nuclei for each spore type, generated by
  #   predict_spore_nucleus.py
  #
  #   nucleus: recorded nucleus data for Microsporidia spores
  #
  # ---------------------------------------------------------------------------
  # Just treat spore nucleus predictions as vectors of 1s and 2s
  # I'll explain this more in lab meeting
  pred_nucleus <- as.numeric(str_extract_all(pred_nucleus, '\\d')[[1]])
  nucleus <- str_remove_all(str_split(nucleus, '; ')[[1]], ' \\(.*')
  
  # writing this as an ifelse() function wasn't working for some reason
  if (length(nucleus) == 1 & nucleus == '') {
    nucleus <- numeric()
  } else {
    nucleus <- as.numeric(nucleus)
  }
  
  tp = get_nucleus_tp(pred_nucleus, nucleus)
  fp = length(pred_nucleus) - tp
  fn = length(nucleus) - tp
  
  return(str_c(tp, fp, fn, sep = ','))
}


nucleus_preds <- read_csv('/home/boognish/Desktop/microsporidia_nlp/results/microsp_spore_nuclei_predictions.csv') %>%
  filter(num_papers < 2, !is.na(pred_nucleus) | !is.na(nucleus),
         nucleus_data_in_text) %>%
  rowwise() %>%
  mutate(pred_nucleus = ifelse(!is.na(pred_nucleus), pred_nucleus, ''),
         nucleus = ifelse(!is.na(nucleus), nucleus, ''),
         tp_fp_fn = get_nucleus_tp_fp_fn(pred_nucleus, nucleus)) %>%
  separate(col= tp_fp_fn, into = c('tp', 'fp', 'fn'), sep = ',') %>%
  mutate(tp = as.integer(tp),
         fp = as.integer(fp),
         fn = as.integer(fn))

# 74.5% precision, 40% recall if consider all data
# 74.5% precision, 66.4% recall if only consider entries where nucleus data is
# in abstract
nucleus_precision <-
  sum(nucleus_preds$tp) / (sum(nucleus_preds$tp) + sum(nucleus_preds$fp))

nucleus_recall <-
  sum(nucleus_preds$tp) / (sum(nucleus_preds$tp) + sum(nucleus_preds$fn))

# to_check <- filter(nucleus_preds, tp < 1 | fp > 0 | fn > 0)