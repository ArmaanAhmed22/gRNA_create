from collections import Counter

import pytest

from gRNA_create.pam import PAM, End


@pytest.mark.parametrize("PAM_1, PAM_2, expected", [
    (PAM(End(3), "TTN"), PAM(End(3), "TTT"), True),
    (PAM(End(5), "TTN"), PAM(End(5), "TTA"), True),
    (PAM(End(3), "TTM"), PAM(End(3), "TTR"), True),
    (PAM(End(3), "TMT"), PAM(End(3), "TKT"), False),
    (PAM(End(3), "TTN"), PAM(End(5), "TTT"), False),
    (PAM(End(3), "KTN"), PAM(End(3), "TTT"), True)
])
def test_pam_overlap(PAM_1: PAM, PAM_2: PAM, expected: bool):
    assert PAM.overlap(PAM_1, PAM_2) is expected


@pytest.mark.parametrize("PAM_1, expected", [
    (PAM(End(3), "TTN"), [PAM(End(3), cur_PAM_seq) for cur_PAM_seq in ["TTT", "TTA", "TTG", "TTC"]]),
    (PAM(End(3), "NTK"),
     [PAM(End(3), cur_PAM_seq) for cur_PAM_seq in ["ATT", "GTT", "CTT", "TTT", "ATG", "GTG", "CTG", "TTG"]])
])
def test_pam_generate_non_ambiguous(PAM_1: PAM, expected: list[PAM]):
    assert Counter(PAM_1.generate_non_ambiguous()) == Counter(expected)
