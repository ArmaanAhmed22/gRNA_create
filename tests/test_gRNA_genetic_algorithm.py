import pandas as pd
import pytest

from gRNA_create.gRNA import gRNA_Factory
from gRNA_create.gRNA_genetic_algorithm import gRNAGroupGA
from gRNA_create.gRNA_scorer import Scorer, TwoPenaltyScorer
from gRNA_create.pam import PAM, End
from gRNA_create.utils import f_score
from tests.conf import positive_seqs, negative_seqs


@pytest.mark.parametrize("gRNA_PAM, gRNA_length, scorer, scoring_metric, expected", [
    (PAM(End(5), "TT"), 14, TwoPenaltyScorer(), f_score(3), 1.0)
])
def test_gRNA_genetic_algorithm(
        gRNA_PAM: PAM,
        gRNA_length: int,
        scorer: Scorer,
        scoring_metric,
        expected: float,
):
    results: pd.DataFrame = gRNA_Factory(gRNA_PAM, gRNA_length, scorer). \
        create_gRNAs(positive_seqs, scoring_metric, genomes_miss=negative_seqs, bind_cutoff=1.0)
    GA_parameters = {
        "num_generations": 200,
        "num_parents_mating": 2,
        "scoring_metric": scoring_metric,
        # "initial_gRNAs" : [gRNA(37,"CUUCCGAUUAUGUUGGCAGG",pam,s)]*20,
        "initial_gRNAs": results,
        "targets": positive_seqs,
        "misses": negative_seqs,
        "mutation_rate": 0.05,
        "binding_cutoff": 1.0,
        "multiplier": 3
    }
    genetic: gRNAGroupGA = gRNAGroupGA(
        **GA_parameters
    )
    genetic.run()
    gen_res: pd.DataFrame = genetic.get_result()
    assert gen_res.sort_values(by=scoring_metric.__name__, ascending=False)[scoring_metric.__name__].iloc[0] == expected
