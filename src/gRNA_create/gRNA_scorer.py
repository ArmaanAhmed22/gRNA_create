import abc
from typing import Dict

from gRNA_create.pam import PAM
from gRNA_create.utils import complement_table, transcribe_table
import pandas as pd


class Scorer:
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def penalty(self, gRNA_spacer: str, target: str, needed_PAM: PAM, current_PAM: PAM) -> float:
        """
        Generates a penalty score between the current gRNA and target.

        Parameters
        ----------
        gRNA_spacer:
            The spacer of the gRNA sequence
        target:
            The target sequence of the gRNA sequence (ideally the target should exactly match the spacer)
        needed_PAM:
            The PAM needed for binding
        current_PAM:
            The current PAM sequence of the target

        Returns
        -------
        A 0.0 to 1.0 penalty score between the gRNA and target
        """
        pass


class TwoPenaltyScorer(Scorer):
    def penalty(self, gRNA_spacer: str, target: str, needed_PAM: PAM, current_PAM: PAM) -> float:
        if not PAM.overlap(needed_PAM, current_PAM):
            return 1.0
        cur_penalty: float = 0
        for i, (nuc_pair_0, nuc_pair_1) in enumerate(zip(gRNA_spacer, target)):
            if nuc_pair_1 != nuc_pair_0:
                cur_penalty += 0.4
            if cur_penalty >= 1:
                return 1
        return cur_penalty


class AaCas12b(Scorer):
    def __init__(self):
        self.penalty_scores = [
            0.954274354,
            0.015904573,
            0.888667992,
            0.085487078,
            0.043737575,
            0.071570577,
            0.083499006,
            0.015904573,
            0.194831014,
            0.966202783,
            0.827037773,
            0.381709742,
            0.115308151,
            0.043737575,
            0.842942346,
            0.932405567,
            0.357852883,
            0.051689861,
            0,
            0.083499006
        ]

    def penalty(self, gRNA_spacer: str, target: str, needed_PAM: PAM, current_PAM: PAM) -> float:
        if not PAM.overlap(needed_PAM, current_PAM):
            return 1.0
        penalty: float = 0
        num_misses: int = 0
        for i, (nuc_pair_0, nuc_pair_1) in enumerate(zip(gRNA_spacer, target)):
            if nuc_pair_1 != nuc_pair_0:
                num_misses += 1
                if num_misses > 1:
                    penalty += max(self.penalty_scores[i], 0.5)
                else:
                    penalty += self.penalty_scores[i]

                if penalty >= 1:
                    return 1.0
        return penalty


class CFDScorer(Scorer):
    def __init__(self):
        self.cfd: pd.DataFrame = pd.read_csv("../tests/cfd.csv", index_col=0)
        self.cfd_dict: Dict[str, Dict[str, float]] = self.cfd.to_dict()

    def penalty(self, gRNA_spacer: str, target: str, needed_PAM: PAM, current_PAM: PAM) -> float:
        if not PAM.overlap(needed_PAM, current_PAM):
            return 1.0

        total_CFD: float = 1

        assert len(gRNA_spacer) == len(target)

        for i, (nuc_pair_0, nuc_pair_1) in enumerate(zip(gRNA_spacer, target)):
            pos = i + 1
            if nuc_pair_0 != nuc_pair_1:
                total_CFD *= self.cfd_dict[str(pos)][f"r{transcribe_table[nuc_pair_0]}:d{complement_table[nuc_pair_1]}"]
        return 1 - total_CFD


class TestScorer(Scorer):
    def penalty(self, gRNA_spacer: str, target: str, needed_PAM: PAM, current_PAM: PAM) -> float:

        if not PAM.overlap(needed_PAM, current_PAM):
            return 1.0

        total_penalty: float = 0

        for i in range(len(gRNA_spacer)):
            if gRNA_spacer[i] != target[i]:
                total_penalty += 0.5
                if total_penalty >= 1.0:
                    return 1.0
        return total_penalty
