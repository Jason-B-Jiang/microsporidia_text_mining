# -----------------------------------------------------------------------------
#
# Organize microsporidia species papers
#
# Jason Jiang - Created: 2022/05/02
#               Last edited: 2022/08/19
#
# Mideo Lab - Microsporidia text mining
#
# Select microsporidia species for analysis and extract first papers describing
# each species.
#
# -----------------------------------------------------------------------------

library(tidyverse)
library(writexl)

################################################################################

main <- function() {
  # Supplemental table S1 from Murareanu et al. 2021 (Microsporidia dataset)
  microsp_data <-
    format_microsp_data(readxl::read_xlsx('../../data/Murareanu_Table_S1.xlsx'))
  
  # Exclude any species coming from these papers (see methods for explanation)
  excluded_papers <- readLines('../../data/excluded_papers.txt')
  
  # get references for first papers of each microsporidia species
  microsp_abstracts <- get_microsp_abstracts(microsp_data, excluded_papers)
  
  # add column to microsp_data for paper abstracts associated w/ each microsporidia
  # species
  microsp_data <- add_abstracts_to_microsp_data(microsp_data, microsp_abstracts)
  
  # for microsporidia species that have multiple papers associated with them,
  # separate these species into muliple rows, 1 row per paper
  #
  # because we've only kept track of abstracts for the first paper for each
  # microsporidia species so far, leave all the columns in these extra rows
  # added as NAs
  microsp_data <- add_rows_for_separate_papers(microsp_data)
  
  write_csv(microsp_data, '../../data/microsporidia_species_and_abstracts.csv')
}

################################################################################

### Helper functions for formatting the Microsporidia dataset from Table S1 of
## Murareanu et al's paper

format_microsp_data <- function(microsp_data) {
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  return(
    microsp_data %>%
      select(Species, `Date Identified`, `Hosts (formatted, no experimental hosts)`,
             `Experimental Hosts`, `Site (formatted)`, `Spore Length Range (µm)`,
             `Spore Width Range (µm)`, `Spore Length Average (µm) (class, condition)`,
             `Spore Width Average (µm) (class, condition)`, `Spore Shape (Class; Condition)`,
             Locality, `Nucleus (formatted) (class, condition)`, `Polar Tubule Length Max (μm)`,
             `Polar Tubule Length Min (μm)`, `Actual Polar Tubule (μm)`,
             `Polar Tubule Coils Range`, `Polar Tubule Coils Average`,
             `Citation data`) %>%
      rename(species = Species,
             dates_identified = `Date Identified`,
             hosts_natural = `Hosts (formatted, no experimental hosts)`,
             hosts_experimental = `Experimental Hosts`,
             infection_site = `Site (formatted)`,
             spore_length_range = `Spore Length Range (µm)`,
             spore_width_range = `Spore Width Range (µm)`,
             spore_length_avg = `Spore Length Average (µm) (class, condition)`,
             spore_width_avg = `Spore Width Average (µm) (class, condition)`,
             spore_shape = `Spore Shape (Class; Condition)`,
             locality = Locality,
             nucleus = `Nucleus (formatted) (class, condition)`,
             # pt = polar tube
             pt_max = `Polar Tubule Length Max (μm)`,
             pt_min = `Polar Tubule Length Min (μm)`,
             pt_avg = `Actual Polar Tubule (μm)`,
             pt_coils_range = `Polar Tubule Coils Range`,
             pt_coils_avg = `Polar Tubule Coils Average`,
             all_references = `Citation data`) %>%
      mutate(hosts = combine_natural_and_experimental_hosts(hosts_natural,
                                                            hosts_experimental),
             year_first_described = get_year_first_identified(dates_identified),
             first_paper_reference = get_first_reference(all_references),
             # remove trailing newlines from references
             all_references = trimws(all_references)) %>%
      select(-dates_identified, -hosts_natural, -hosts_experimental)
  )
}


combine_natural_and_experimental_hosts <- Vectorize(function(natural_hosts,
                                                             experimental_hosts) {
  # ---------------------------------------------------------------------------
  # 
  # ---------------------------------------------------------------------------
  if (is.na(natural_hosts)) {
    return(NA)
  }
  
  if (is.na(experimental_hosts)) {
    return(natural_hosts)
  }
  
  
  return(str_c(natural_hosts, experimental_hosts, sep = '; '))
}, vectorize.args = c('natural_hosts', 'experimental_hosts'))


get_year_first_identified <- Vectorize(function(years) {
  # ---------------------------------------------------------------------------
  # Get the first year a Microsporidia species was described in, as an integer
  #
  # Input:
  #   years: entry from the 'Date Identified (year)' column from microsporidia
  #          species dataset
  # ---------------------------------------------------------------------------
  return(as.integer(str_remove(str_split(years, '; ')[[1]][1], ' \\(.+')))
})


get_first_reference <- Vectorize(function(ref) {
  # ---------------------------------------------------------------------------
  # Get reference for the first paper describing a particular Microsporidia
  # species.
  #
  # Input:
  #   ref: entry from References column of Microsporidia dataset
  # ---------------------------------------------------------------------------
  return(str_remove(trimws(str_split(ref, '\n')[[1]][1]), '^\\d\\. '))
})

################################################################################

## Helper functions for adding and formatting paper abstracts associated with
## each microsporidia species

get_microsp_abstracts <- function(microsp_data, excluded_papers) {
  # ---------------------------------------------------------------------------
  # 
  # ---------------------------------------------------------------------------
  if (file.exists('../../data/manually_collect_abstracts.xlsx')) {
    microsp_abstracts <- readxl::read_xlsx('../../data/manually_collect_abstracts.xlsx')
    
    if (all(is.na(microsp_abstracts$abstract))) {
      stop("Please manually add paper abstracts to data/manually_collect_abstracts.xlsx, then rerun this script.")
    }
    
    return(format_abstracts(microsp_abstracts))
    
  } else {
    microsp_abstracts <- select_microsporidia_for_analysis(microsp_data, excluded_papers)
    write_xlsx(microsp_abstracts, '../../data/manually_collect_abstracts.xlsx')
    
    stop("Please manually add paper abstracts to data/manually_collect_abstracts.xlsx, then rerun this script.")
  }
}


select_microsporidia_for_analysis <- function(microsp_data, excluded_papers) {
  # ---------------------------------------------------------------------------
  # To select Microsporidia species to include for text mining, exclude species
  # that come before 1977 (these species are described in the Sprague book)
  #
  # Also, add columns for the paper abstracts for each microsporidium, as
  # well as a column indicating whether the full paper for the microsporidium
  # is in English or not.
  #
  # Args:
  #   microsp_data: dataframe of Table S1 from Murareanu et al.'s paper
  #
  #   excluded_papers: vector of paper titles, where we want to exclude
  #   microsporidia coming from these papers from text mining.
  #
  # ---------------------------------------------------------------------------
  return(
    microsp_data %>%
      mutate(first_paper_title = NA,
             abstract = NA,
             notes = NA,
             # Is the paper in a foreign language?
             foreign = NA) %>%
      # filter out cases where year first described is unknown or ambiguous
      # (NA in the microsporidia dataset for Date Identified)
      filter(!is.na(year_first_described),
             # Filter microsporidia species to species from 1977 to 2021
             # Species from before 1977 come from Sprague book (see methods)
             year_first_described >= 1977) %>%
      rowwise() %>%
      filter(!any(str_detect(all_references, excluded_papers))) %>%
      select(species, year_first_described, first_paper_reference,
             first_paper_title, abstract, notes, foreign) %>%
      arrange(year_first_described)
  )
}


format_abstracts <- function(microsp_abstracts) {
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  return(
    microsp_abstracts %>%
      # fix weird issue with true/false being turned into dates by xlsx format
      mutate(foreign = ifelse(!is.na(foreign), TRUE, FALSE),
             first_paper_title = clean_text(first_paper_title),
             abstract = clean_text(abstract),
             # create new column combining paper title and abstract
             title_abstract = ifelse(!is.na(abstract),
                                     str_c(first_paper_title, '. ', abstract),
                                     first_paper_title)) %>%
      select(-notes)
  )
}


clean_text <- Vectorize(function(text) {
  # ---------------------------------------------------------------------------
  # Replace tabs and newlines with spaces, remove excess + trailing whitespacee
  # from a string.
  #
  # Input:
  #   text: paper title or abstract
  # ---------------------------------------------------------------------------
  trimws(str_replace_all(
    str_replace_all(str_replace_all(text, '\t', ' '), '\n', ' '),
    ' +',
    ' '))
})


add_abstracts_to_microsp_data <- function(microsp_data, microsp_abstracts) {
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  return(
    merge(x = microsp_abstracts, y = microsp_data,
          by = 'species', all = TRUE) %>%
      filter(!is.na(first_paper_title)) %>%
      select(-year_first_described.y, -first_paper_reference.y) %>%
      rename(year_first_described = year_first_described.x,
             first_paper_reference = first_paper_reference.x) %>%
      select(species, year_first_described, first_paper_reference,
             first_paper_title, abstract, title_abstract, foreign, hosts,
             everything())
  )
}


add_rows_for_separate_papers <- function(microsp_data) {
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  columns_of_interest <- c('species', 'year_first_described', 'first_paper_reference',
                           'all_references', 'num_papers')
  
  columns_to_clear <-
    colnames(microsp_data)[!(colnames(microsp_data) %in% columns_of_interest)]
  
  # add column for number of papers each microsporidia species has, and put each
  # paper in individual rows
  microsp_data <- microsp_data %>%
    rowwise() %>%
    mutate(num_papers = length(str_split(all_references, '\n')[[1]])) %>%
    separate_rows(all_references, sep = '\n') %>%
    mutate(all_references = trimws(all_references))
  
  for (i in 1 : nrow(microsp_data)) {
    # if num_papers > 1:
    #   set columns of non-interest to NA
    #   add numbering to species w/ refs numbering
    #   add paper_ref
    if (microsp_data$num_papers[i] > 1) {
      microsp_data[i, 'species'] <-
        str_c(microsp_data$species[i],
              ' (',
              str_extract(microsp_data$all_references[i], '^\\d+(?=\\.)'),
              ')') 
      
      # if not entry for first paper, clear all columns for COLUMNS_TO_CLEAR an
      if (!str_detect(microsp_data$all_references[i], '^1\\.')) {
        microsp_data[i, columns_to_clear] <-
          as.list(rep(NA, length(columns_to_clear)))
      }
    }
  }
  
  return(microsp_data)
}

################################################################################

main()