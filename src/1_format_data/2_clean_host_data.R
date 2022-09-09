# -----------------------------------------------------------------------------
#
# Clean microsporidia name + host data for predictions
#
# Jason Jiang - Created: 2022/07/29
#               Last edited: 2022/09/09
#
# Mideo Lab - Microsporidia text mining
#
# Clean up microsporidia/host name data for predicting with rule-based methods,
# and add info about where each microsporidia/host name are found in text.
#
# -----------------------------------------------------------------------------

library(tidyverse)

################################################################################

main <- function() {
  microsp_host_data <- format_host_data(
    read_csv('../../data/microsporidia_species_and_abstracts.csv', show_col_types = F)
    )
  
  # check if manual corrections file made
  # if yes, check if corrections made
  # if yes, implement changes into microsp_host_data
  # else, prompt user to make the manual corrections
  #
  # if file not existing, make the file and prompt user to fill it out
  if (!file.exists('../../data/manually_corrected_hosts_and_microsp.csv')) {
    to_manually_correct <- microsp_host_data %>%
      filter(!is.na(hosts_not_in_text) | !is.na(microsp_not_in_text))
    
    write_csv(to_manually_correct, '../../data/manually_corrected_hosts_and_microsp.csv')
    stop('Please manually correct recorded microsporidia/host names in the file created.')
    
  } else {
    manually_corrected_names <-
      read_csv('../../data/manually_corrected_hosts_and_microsp.csv')
    
    if (all(is.na(manually_corrected_names$hosts_not_in_text_corrected)) &
        all(is.na(manually_corrected_names$microsp_not_in_text_corrected))) {
      stop('Please manually correct recorded microsporidia/host names in the file created.')
      
    } else {
      microsp_host_data <- implement_manual_corrections(microsp_host_data,
                                                        manually_corrected_names) %>%
        rowwise() %>%
        mutate(microsp_in_text_matches = get_species_matches_in_text(microsp_in_text,
                                                                     title_abstract),
               hosts_in_text_matches = get_species_matches_in_text(hosts_in_text,
                                                                   title_abstract)) %>%
        group_by(title_abstract) %>%
        mutate(species = str_c(str_replace_na(species),
                               collapse = ' || '),
               species_cleaned = str_c(str_replace_na(species_cleaned),
                                       collapse = ' || '),
               microsp_in_text = str_c(str_replace_na(microsp_in_text),
                                       collapse = ' || '),
               hosts_in_text = str_c(str_replace_na(hosts_in_text),
                                     collapse = ' || '),
               microsp_in_text_matches = str_c(str_replace_na(microsp_in_text_matches),
                                               collapse = ' || '),
               hosts_in_text_matches = str_c(str_replace_na(hosts_in_text_matches),
                                             collapse = ' || ')) %>%
        select(species, species_cleaned, title_abstract,
               microsp_in_text, hosts_in_text, microsp_in_text,
               microsp_in_text_matches, hosts_in_text_matches) %>%
        distinct(.keep_all = T)
    }
  }
  
  write_csv(microsp_host_data, '../../data/formatted_host_microsp_names.csv')
}

################################################################################

## Helper functions

format_host_data <- function(microsp_data) {
  # ---------------------------------------------------------------------------
  # For the dataframe of microsporidia species data + paper abstracts from
  # 1_add_paper_abstracts_for_microsporidia.R, select relevant columns and
  # clean up recorded microsporidia/host names.
  #
  # ---------------------------------------------------------------------------
  return(
    microsp_data %>%
      # only consider the first papers for each species for now
      filter(!str_detect(species, '\\([2-9]\\)$')) %>%
      # strip away parenthesized text from microsporidia + host names
      mutate(species_cleaned = clean_species_names(species),
             hosts_cleaned = clean_species_names(hosts)) %>%
      select(species, species_cleaned, first_paper_title, abstract,
             title_abstract, hosts, hosts_cleaned, num_papers) %>%
      rowwise() %>%
      mutate(hosts_not_in_text = get_species_not_in_text(hosts_cleaned, title_abstract),
             microsp_not_in_text = get_species_not_in_text(species_cleaned, title_abstract),
             microsp_not_in_text_corrected = NA,
             hosts_not_in_text_corrected = NA)
  )
}


clean_species_names <- Vectorize(function(species) {
  # ---------------------------------------------------------------------------
  # For a semi-colon separated string of microsporidia/host names, remove any
  # parenthesized text or numbers
  #
  #   Ex: Nosema melgethi (= Nosema mela) -> Nosema melgethi
  #   Ex: Encephalitozoon sp. 7 -> Encephalitozoon sp.
  #   Ex: Aedes punctor (?); Culex pipeins (= Culex pipens) ->
  #       Aedes punctor; Culex pipeins
  #
  # ---------------------------------------------------------------------------
  species <- trimws(str_remove(str_split(species, '; ')[[1]], ' ?\\(.+\\)'))
  return(str_c(str_remove(species, ' \\d+$'), collapse = '; '))
})


get_abbreviated_species_name <- function(species) {
  # ---------------------------------------------------------------------------
  # Get the abbreviated version of a species name.
  #
  # Ex: Culex pipiens -> C. pipiens
  # Ex: Culex pipiens pipiens -> C. p. pipiens
  # ---------------------------------------------------------------------------
  species <- str_split(species, ' +')[[1]]
  
  if (length(species) == 1) {
    return(species)
  }
  
  # ugly string concat operation
  str_c(
    str_c(
      sapply(species[1 : length(species) - 1], function(s) {str_c(substr(s, 1, 1), '.')}),
      collapse = ' '),
    species[length(species)], sep = ' '
  )
}


get_species_not_in_text <- function(species, txt) {
  # ---------------------------------------------------------------------------
  # Return whether a species appears in a string, txt, or not, checking for
  # both its full and abbreviated name.
  #
  # Arguments:
  #   species: semi-colon separated string of species names
  #   txt: some text (i.e: paper abstract) to look for mentions of species in
  # ---------------------------------------------------------------------------
  if (is.na(txt) | is.na(species)) {
    # no species or no text to check
    return(NA)
  }
  
  # split string of semi-colon separated species into a vector with each
  # individual species
  species <- str_split(species, '; ')[[1]]
  species_not_in_text <- character()
  
  # look for the full + abbreviated species names as lowercase substrings in txt
  for (sp in species) {
    if (!str_detect(tolower(txt), tolower(fixed(sp))) & 
        !str_detect(tolower(txt), tolower(fixed(get_abbreviated_species_name(sp))))) {
      species_not_in_text <- c(species_not_in_text, sp)
    }
  }
  
  if (length(species_not_in_text) == 0) {
    # all species are found in txt, so return NA
    return(NA)
  }
  
  # otherwise, return semi-colon separated string of all species names not
  # appearing in txt
  return(str_c(species_not_in_text, collapse = '; '))
}


implement_manual_corrections <- function(microsp_host_data,
                                         manually_corrected_names) {
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  host_corrections <- new.env()
  microsp_corrections <- new.env()
  
  Map(
    function(species, corrected_hosts, corrected_microsp) {
      host_corrections[[species]] <- corrected_hosts
      microsp_corrections[[species]] <- corrected_microsp
    },
    manually_corrected_names$species,
    manually_corrected_names$hosts_not_in_text_corrected,
    manually_corrected_names$microsp_not_in_text_corrected
  )
  
  return(
    microsp_host_data %>%
      rowwise() %>%
      mutate(microsp_not_in_text_corrected = ifelse(!is_null(microsp_corrections[[species]]),
                                                    microsp_corrections[[species]],
                                                    NA),
             hosts_not_in_text_corrected = ifelse(!is_null(host_corrections[[species]]),
                                                  host_corrections[[species]],
                                                  NA)) %>%
      ungroup() %>%
      mutate(microsp_in_text = ifelse(!is.na(microsp_not_in_text),
                                      microsp_not_in_text_corrected,
                                      species_cleaned),
             hosts_in_text = resolve_recorded_hosts(hosts_cleaned,
                                                    hosts_not_in_text,
                                                    hosts_not_in_text_corrected))
  )
}


resolve_recorded_hosts <- Vectorize(function(hosts, hosts_not_in_text, hosts_corrected) {
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  if (is.na(hosts_not_in_text)) {
    return(hosts)
    
  } else {
    hosts <- str_split(hosts, '; ')[[1]]
    hosts_not_in_text <- str_split(hosts_not_in_text, '; ')[[1]]
    hosts_corrected <- str_split(hosts_corrected, '; ')[[1]]
    
    hosts <- hosts[!(hosts %in% hosts_not_in_text)]
    
    if (all(is.na(hosts_corrected))) {
      return(ifelse(length(hosts) > 0,
                    str_c(hosts, collapse = '; '),
                    NA))
    }
    
    return(str_c(c(hosts, hosts_corrected), collapse = '; '))
  }
}, vectorize.args = c('hosts', 'hosts_not_in_text', 'hosts_corrected'))


get_species_matches_in_text <- function(species, text) {
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  # get abbreviations for species names and join together w/ full names
  # get indices of species names occurrences in text
  if (is.na(species)) {
    return(NA)
  }
  
  species <- str_split(species, '; ')[[1]]
  species <- tolower(unname(
    c(species, sapply(species, function(s) {get_abbreviated_species_name(s)}))
  ))
  
  matches <- str_locate_all(tolower(text), fixed(species))
  matches <- matches[lapply(matches, length) > 0]
  return(
    str_c(
      as.character(lapply(matches, function(x) {str_c(x[1], x[2], sep = '-')})),
    collapse = '; ')
  )
}

################################################################################

main()