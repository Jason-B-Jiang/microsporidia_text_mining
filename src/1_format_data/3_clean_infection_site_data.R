library(tidyverse)

microsp_data <- read_csv('../../data/microsporidia_species_and_abstracts.csv', show_col_types = F)

microsp_infection_sites <- microsp_data %>%
  select(species, title_abstract, hosts, infection_site) %>%
  filter(!is.na(title_abstract)) %>%
  mutate(species = str_remove(species, ' \\(.+'))

clean_host_data <- Vectorize(function(hosts) {
  hosts <- str_split(hosts, '; ')[[1]]
  hosts <- sapply(hosts, function(h) {str_remove(h, ' ?\\(.+')})
  return(str_c(hosts, collapse = '; '))
})

microsp_infection_sites <- microsp_infection_sites %>%
  mutate(hosts = clean_host_data(hosts))

multiple_hosts <- microsp_infection_sites %>%
  filter(str_detect(hosts, ';'), !is.na(infection_site)) %>%
  mutate(infection_site_corrected = NA,
         hosts_corrected = NA)

write_csv(multiple_hosts, 'multiple_host_infections_corrected.csv')