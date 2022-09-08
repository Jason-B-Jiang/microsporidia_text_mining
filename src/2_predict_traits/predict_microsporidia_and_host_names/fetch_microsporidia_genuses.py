# -----------------------------------------------------------------------------
#
# Fetch Microsporidia genus names
#
# Jason Jiang - Created: 2022/05/25
#               Last edited: 2022/05/25
#
# Mideo Lab - Microsporidia text mining
#
# Retrieve all Microsporidia genus names from NCBI's taxonomy database, for use
# in predicting microsporidia species names from texts.
#
# -----------------------------------------------------------------------------

from Bio import Entrez
import os

################################################################################

## Global variables

EMAIL = 'jasonbitan.jiang@mail.utoronto.ca'   # fill in your email here
TAXONKIT_DATA = os.path.abspath('../../../bin')  # directory where NCBI database dump for
                                              # Taxonkit is stored
OUT_DIR = os.path.abspath('../../../data/microsp_genuses')  # path for saving script outputs

################################################################################

# Get TaxID of microsporidia in NCBI taxonomy database, so we can use Taxonkit
# later to retrieve all microsporidia genuses
Entrez.email = EMAIL
record = Entrez.read(Entrez.esearch(db='Taxonomy', term='Microsporidia'))
microsporidia_taxid = record['IdList'][0]

################################################################################

os.mkdir(OUT_DIR)

# Initialize tsv file for storing microsporidia genuses
os.system(
    f"echo -e 'tax_id\tname\ttax_rank' > {OUT_DIR}/microsporidia_genuses.tsv"
)

# Use Taxonkit to find all genuses associated with microsporidia TaxID, and append
# to the tsv initialized above

# Make sure you've installed Taxonkit prior to running this

# In my case, I installed it in Conda with 'conda install -c bioconda taxonkit'
# I selected the Python interpreter for the Conda environment where I installed
# Taxonkit in VScode before executing this script in VScode
os.system(
    f"taxonkit list --ids {microsporidia_taxid} --data-dir {TAXONKIT_DATA} | \
        awk '{{$1=$1;print}}' | \
            cat | \
                taxonkit filter -E genus --data-dir {TAXONKIT_DATA} | \
                    taxonkit lineage -r -n -L --data-dir {TAXONKIT_DATA} >> \
                        {OUT_DIR}/microsporidia_genuses.tsv"
)
