import pandas as pd
import pytest
from Bio import SeqIO

from gRNA_create.gRNA import gRNA_Factory
from gRNA_create.gRNA_scorer import Scorer, TwoPenaltyScorer
from gRNA_create.pam import PAM, End
from gRNA_create.utils import f_score


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
    with open("tests/test_sequences.fasta", "r") as h:
        positive_seqs = [str(record.seq) for record in SeqIO.parse(h, format="fasta")]
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
    with open("tests/test_sequences.fasta", "r") as h:
        positive_seqs = [str(record.seq) for record in SeqIO.parse(h, format="fasta")]
        results: pd.DataFrame = gRNA_Factory(gRNA_PAM, gRNA_length, scorer). \
            create_gRNAs_with_reverse_complement(positive_seqs, scoring_metric)
        print(results)
        assert results.shape[0] == expected
