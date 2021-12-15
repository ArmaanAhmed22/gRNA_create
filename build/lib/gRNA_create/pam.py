from __future__ import annotations

from typing import List

from Bio.Data.IUPACData import ambiguous_dna_values
from enum import Enum
from gRNA_create.utils import ambiguous_code
from itertools import product


class End(Enum):
    """
    An enum containing the possible prime ends (5' end and 3' end)
    """
    PRIME_5 = 5
    PRIMER_3 = 3


class PAM:
    ambiguous_dna_values_set = {k: set(v) for k, v in ambiguous_dna_values.items()}

    def __init__(self, location: End, sequence: str):
        """
        A PAM is usually necessary for Cas-gRNA binding to a target sequence. This object encodes a PAM

        :param location: Specifies which side of the spacer the PAM is on
        :param sequence: Specifies the actual PAM sequence (can contain degenerate nucleotides)
        """
        self.location = location
        self.sequence = sequence
        self.length = len(sequence)

    def generate_non_ambiguous(self) -> List[PAM]:
        """
        Generates all non-degenerate PAMs that are subsets of this PAM

        :return: List of non-degenerate PAMs
        """

        pams = [PAM(self.location, "".join(possible)) for possible in
                product(*[PAM.ambiguous_dna_values_set[nuc] for nuc in self.sequence])]
        return pams

    @staticmethod
    def overlap(pam1: PAM, pam2: PAM) -> bool:
        """
        Checks whether "pam1" and "pam2" overlap, meaning whether they share any non-degenerate PAMs

        :param pam1: The first PAM
        :param pam2: The second PAM
        :return: Whether "pam1" and "pam2" overlap
        """
        if pam1.location != pam2.location:
            return False
        for i in range(pam1.length):
            try:
                if ambiguous_code[pam1[i]] & ambiguous_code[pam2[i]] == 0:
                    return False
            except Exception:
                print(pam1, "    ", pam2)
                raise Exception()
        return True

    def __hash__(self):
        return hash((self.location, self.sequence))

    def __eq__(self, other):
        if isinstance(other, PAM):
            return self.location == other.location and self.sequence == other.sequence
        return False

    def __str__(self) -> str:
        return self.sequence.__str__()

    def __getitem__(self, key: int) -> str:
        return self.sequence.__getitem__(key)
