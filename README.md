# gRNA Creator
## Description
Here, I present a tool to create gRNAs for a myriad of different Cas enzyme. gRNAs have the option to also be designed to be specific against certain targets. This is especially useful for the diagnostic application of gRNAs, where it is imperative to discriminate between wild-type and mutant targets

## Installation
1) `git clone https://github.com/ArmaanAhmed22/gRNA_create`
2) `cd gRNA_create`
3) `python -m pip install . -r requirements.txt`

## Usage
Creating gRNAs against target:
```python
from gRNA_create.gRNA import gRNA_Factory
from gRNA_create.pam import *
from gRNA_create.gRNA_scorer import CFDScorer
from gRNA_create.utils import sensitivity

pam = PAM(End(3),"NGG") #PAM for SpCas9
scorer = CFDScorer() #The scoring algorithm for target-gRNA mismatches
gRNA_length = 20

factory = gRNA_Factory(pam,gRNA_length,scorer)

gRNA_DataFrame = factory.create_gRNAs(
    genomes_target = "path/to/targets/dir",
    scoring_metric = sensitivity, # metric to measure how "good" any given gRNA is
    genomes_miss = "path/to/misses/dir", # [OPTIONAL]
    pam_minimum_prevalence = 0.8 # [OPTIONAL] | The PAM prevalence in targets needed to keep any gRNAs
)
```
Creating 
![Tests](https://github.com/ArmaanAhmed22/gRNA_create/actions/workflows/tests.yaml/badge.svg)
