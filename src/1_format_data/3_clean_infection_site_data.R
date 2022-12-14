library(tidyverse)

################################################################################

clean_host_data <- Vectorize(function(hosts) {
  hosts <- str_split(hosts, '; ')[[1]]
  hosts <- sapply(hosts, function(h) {str_remove(h, ' \\(.+')})
  return(str_c(hosts, collapse = '; '))
})

microsp_infection_sites <- read_csv('../../data/microsporidia_species_and_abstracts.csv',
                                    show_col_types = F) %>%
  select(species, title_abstract, hosts, infection_site) %>%
  filter(!is.na(title_abstract)) %>%
  mutate(hosts = clean_host_data(hosts))

################################################################################

merge_recorded_and_corrected <- Vectorize(function(recorded, absent, corrected) {
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  if (is.na(recorded)) {
    recorded <- character(0)
  } else {
    recorded <- str_split(recorded, '; ')[[1]]
  }
  
  if (is.na(absent)) {
    absent <- character(0)
  } else {
    absent <- str_split(absent, '; ')[[1]]
  }
  
  if (is.na(corrected)) {
    corrected <- character(0)
  } else {
    corrected <- str_split(corrected, '; ')[[1]]
  }
  
  merged <- c(recorded[!(recorded %in% absent)], corrected)
  if (length(merged) < 1) {
    return(NA)
  }
  
  return(str_c(merged, collapse = '; '))
}, vectorize.args = c('recorded', 'absent', 'corrected'))

manually_corrected_species <- read_csv('../../data/manually_corrected_hosts_and_microsp.csv') %>%
  select(species, species_cleaned, microsp_not_in_text, microsp_not_in_text_corrected,
         title_abstract, hosts_cleaned, hosts_not_in_text,
         hosts_not_in_text_corrected) %>%
  mutate(species_final = ifelse(!is.na(microsp_not_in_text),
                                microsp_not_in_text_corrected,
                                species_cleaned),
         hosts_final = merge_recorded_and_corrected(hosts_cleaned,
                                                    hosts_not_in_text,
                                                    hosts_not_in_text_corrected))

species_corrections <- new.env()
for (i in 1 : nrow(manually_corrected_species)) {
  species_corrections[[manually_corrected_species$species[i]]] <-
    c(
      unname(manually_corrected_species$species_final[i]),
      unname(manually_corrected_species$hosts_final[i])
    )
}

################################################################################

microsp_infection_sites <- microsp_infection_sites %>%
  rowwise() %>%
  mutate(species_corrected = ifelse(species %in% manually_corrected_species$species,
                                    species_corrections[[species]][1],
                                    str_remove(species, ' (\\d+|\\(.+\\))$')),
         hosts_corrected = ifelse(species %in% manually_corrected_species$species,
                                  species_corrections[[species]][2],
                                  hosts)) %>%
  select(species_corrected, hosts_corrected, infection_site, title_abstract, everything())

################################################################################

# clean-up and shit

parenthesized_info_to_exclude <- c('main', 'except', '\\d+', 'of ', 'in ',
                                   'primary', 'larva', 'adult', 'male', 'spore',
                                   'infect')

get_parenthesized_info <- function(infection_site) {
  infection_site <- str_split(infection_site, '; ')[[1]]
  parenthesized_info <- sapply(infection_site, function(s) {str_extract(s, '(?<=\\().+(?=\\))')})
  
  parenthesized_info <- parenthesized_info[!is.na(parenthesized_info)]
  
  parenthesized_info_filtered <- c()
  for (info in parenthesized_info) {
    info <- str_split(info, ', ')[[1]]
    for (subinfo in info) {
      if (!any(str_detect(subinfo, parenthesized_info_to_exclude))) {
        parenthesized_info_filtered <- append(parenthesized_info_filtered, subinfo)
      }
    }
  }
  
  return_str = str_c(parenthesized_info_filtered, collapse = '; ')
  return(ifelse(return_str == '', NA, return_str))
}


remove_parenthesized_info <- function(infection_site) {
  infection_site <- str_split(infection_site, '; ')
  infection_site <- sapply(infection_site, function(s) {str_remove(s, ' \\(.+')})
  
  return_str = str_c(infection_site, collapse = '; ')
  return(ifelse(return_str == '', NA, return_str))
}


get_entries_not_in_text <- Vectorize(function(entries, text) {
  entries <- str_split(entries, '; ')[[1]]
  
  not_in_text <- entries[!str_detect(tolower(text), tolower(entries))]
  
  return_str = str_c(not_in_text, collapse = '; ')
  return(ifelse(return_str == '', NA, return_str))
}, vectorize.args = c('entries', 'text'))


microsp_infection_sites <- microsp_infection_sites %>%
  mutate(infection_site_parenthesized = get_parenthesized_info(infection_site),
         infection_site = remove_parenthesized_info(infection_site),
         parenthesized_not_in_text = get_entries_not_in_text(infection_site_parenthesized,
                                                             title_abstract),
         infection_site_not_in_text = get_entries_not_in_text(infection_site,
                                                              title_abstract),
         parenthesized_corrected = NA,
         infection_site_corrected = NA) %>%
  select(species, title_abstract, species_corrected, hosts_corrected, infection_site,
         infection_site_parenthesized, infection_site_not_in_text,
         parenthesized_not_in_text, infection_site_corrected,
         parenthesized_corrected) %>%
  filter(!(species_corrected %in% c('Microsporidium 1', 'Microsporidium 2')))

# TODO - add new species to check from additional condition
to_check <- microsp_infection_sites %>%
  filter(!is.na(parenthesized_not_in_text) | !is.na(infection_site_not_in_text) | str_detect(hosts_corrected, ';'))

write_csv(to_check, 'manually_corrected_infection_sites.csv')

################################################################################

## Incorporate corrected infection sites into microsp_infection_sites

corrected <- read_csv('manually_corrected_infection_sites_CORRECTED.csv')

get_final_infection_site <- function(infection_site,
                                     infection_site_parenthesized,
                                     infection_site_not_in_text,
                                     parenthesized_not_in_text,
                                     infection_site_corrected,
                                     parenthesized_corrected) {
  # convert each argument into a vector, so we can do set operations
  # also make everything lowercase so matches are case-insensitive
  ifelse(!is.na(infection_site),
         infection_site <- tolower(str_split(infection_site, '; ')[[1]]),
         infection_site <- character(0))
  
  ifelse(!is.na(infection_site_parenthesized),
         infection_site_parenthesized <- tolower(str_split(infection_site_parenthesized, '; ')[[1]]),
         infection_site_parenthesized <- character(0))
  
  ifelse(!is.na(infection_site_not_in_text),
         infection_site_not_in_text <- tolower(str_split(infection_site_not_in_text, '; ')[[1]]),
         infection_site_not_in_text <- character(0))
  
  ifelse(!is.na(parenthesized_not_in_text),
         parenthesized_not_in_text <- tolower(str_split(parenthesized_not_in_text, '; ')[[1]]),
         parenthesized_not_in_text <- character(0))
  
  ifelse(!is.na(infection_site_corrected),
         infection_site_corrected <- tolower(str_split(infection_site_corrected, '; ')[[1]]),
         infection_site_corrected <- character(0))
  
  ifelse(!is.na(parenthesized_corrected),
         parenthesized_corrected <- tolower(str_split(parenthesized_corrected, '; ')[[1]]),
         parenthesized_corrected <- character(0))
  
  all_sites <- union(infection_site, infection_site_parenthesized)
  not_in_text <- union(infection_site_not_in_text, parenthesized_not_in_text)
  in_text <- union(infection_site_corrected, parenthesized_corrected)
  
  # subtract sites not found in text from all sites, then add back in corrected
  # sites that are found in the text for our final set of infection sites
  final_sites <- union(setdiff(all_sites, not_in_text), in_text)
  
  if (length(final_sites) < 1) {
    return(NA)
  }
  
  return(str_c(final_sites, collapse = '; '))
}

corrected <- corrected %>%
  rowwise() %>%
  # if we have multiple hosts recorded for a species, just replace the
  # entry with infection_site_corrected
  mutate(infection_site_final = ifelse(str_detect(hosts_corrected, '; ') & !is.na(hosts_corrected),
                                       tolower(infection_site_corrected),
                                       get_final_infection_site(infection_site,
                                                                infection_site_parenthesized,
                                                                infection_site_not_in_text,
                                                                parenthesized_not_in_text,
                                                                infection_site_corrected,
                                                                parenthesized_corrected)))

# create hashmap/dictionary of microsporidia species and their corrected
# infection sites
corrected_dict <- new.env()
Map(function(species, corrected_sites) {corrected_dict[[species]] <- corrected_sites},
    corrected$species,
    corrected$infection_site_final)

# apply corrections to microsp_infection_sites
microsp_infection_sites <- microsp_infection_sites %>%
  rowwise() %>%
  mutate(infection_site_corrected = ifelse(!is.null(corrected_dict[[species_corrected]]),
                                       corrected_dict[[species_corrected]],
                                       tolower(infection_site))) %>%
  select(species, title_abstract, species_corrected, hosts_corrected,
         infection_site, infection_site_corrected)

# save corrections as csv
write_csv(microsp_infection_sites, '../../data/microsp_infection_sites.csv')