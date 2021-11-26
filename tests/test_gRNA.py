import pandas as pd
import pytest
from gRNA_create.gRNA import gRNA_Factory
from gRNA_create.gRNA_scorer import Scorer, TwoPenaltyScorer
from gRNA_create.pam import PAM, End
from gRNA_create.utils import f_score, sensitivity
from tests.conf import negative_seqs, positive_seqs, positive_seqs_large


@pytest.mark.parametrize("gRNA_PAM, gRNA_length, scorer, scoring_metric, expected", [
    (TTN := PAM(End(3), "TTN"), 20, TwoPenaltyScorer(), f_score(3), 1),
    (TTN, 28, TwoPenaltyScorer(), f_score(3), 0),
    (PAM(End(5), "TT"), 14, TwoPenaltyScorer(), f_score(3), 3),
    (PAM(End(5), "AGGTTAGCTAGACATGACAAATTGAACAAGTACACG"), 1, TwoPenaltyScorer(), f_score(3), 2)
])
def test_factory_create_gRNAs_positive(
        gRNA_PAM: PAM,
        gRNA_length: int,
        scorer: Scorer,
        scoring_metric,
        expected: int,
):
    results: pd.DataFrame = gRNA_Factory(gRNA_PAM, gRNA_length, scorer).create_gRNAs(positive_seqs, scoring_metric)
    assert results.shape[0] == expected


@pytest.mark.parametrize("gRNA_PAM, gRNA_length, scorer, scoring_metric, expected", [
    (TTN := PAM(End(3), "TTN"), 20, TwoPenaltyScorer(), f_score(3), 1),
    (TTN, 28, TwoPenaltyScorer(), f_score(3), 0),
    (PAM(End(5), "TT"), 14, TwoPenaltyScorer(), f_score(3), 7),
    (PAM(End(5), "AGGTTAGCTAGACATGACAAATTGAACAAGTACACG"), 1, TwoPenaltyScorer(), f_score(3), 2)
])
def test_factory_create_gRNAs_positive_and_rev(
        gRNA_PAM: PAM,
        gRNA_length: int,
        scorer: Scorer,
        scoring_metric,
        expected: int,
):
    results: pd.DataFrame = gRNA_Factory(gRNA_PAM, gRNA_length, scorer). \
        create_gRNAs_with_reverse_complement(positive_seqs, scoring_metric)
    print(results)
    assert results.shape[0] == expected


@pytest.mark.parametrize("gRNA_PAM, gRNA_length, scorer, scoring_metric, expected", [
    (PAM(End(5), "TT"), 14, TwoPenaltyScorer(), f_score(3), "AGCTAGACATGACA")
])
def test_factory_create_gRNAs_pos_neg(
        gRNA_PAM: PAM,
        gRNA_length: int,
        scorer: Scorer,
        scoring_metric,
        expected: str,
):
    results: pd.DataFrame = gRNA_Factory(gRNA_PAM, gRNA_length, scorer). \
        create_gRNAs(positive_seqs, scoring_metric, genomes_miss=negative_seqs)
    assert results.sort_values(by=scoring_metric.__name__, ascending=False).gRNA.iloc[0].spacer_eq(expected)


@pytest.mark.parametrize("gRNA_PAM, gRNA_length, scorer, scoring_metric", [
    (PAM(End(5), "TT"), 14, TwoPenaltyScorer(), sensitivity)
])
def test_factory_create_gRNAs_pos_large(
        gRNA_PAM: PAM,
        gRNA_length: int,
        scorer: Scorer,
        scoring_metric,
):
    gRNA_Factory(gRNA_PAM, gRNA_length, scorer). \
        create_gRNAs(positive_seqs_large, scoring_metric)
