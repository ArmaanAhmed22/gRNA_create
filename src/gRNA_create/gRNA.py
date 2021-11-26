from typing import Callable, Union, List, Dict

from Bio.Seq import Seq

from gRNA_create.gRNA_scorer import Scorer
from gRNA_create.utils import get_sequences_from_dir, complement_table, ConfusionMatrix
from multiprocessing import Pool
import multiprocessing
from gRNA_create.pam import PAM, End
import pandas as pd
from collections import Counter
from tqdm import tqdm
from multiprocessing.pool import Pool as PoolType

from colored import attr

import sys
import traceback


class gRNA(Seq):
    _pool: PoolType

    def __init__(self, position: int, sequence: str, pam: PAM, scorer: Scorer):
        """
        A class representing a gRNA

        :param position: Position starting from the 5' end of the gRNA relative to the target sequences
        :param sequence: The variable region of the gRNA
        :param pam: The PAM of this gRNA
        :param scorer: The scoring algorithm used to penalize mismatches
        """
        super().__init__(sequence.replace("T", "U"))
        self.scorer: Scorer = scorer
        self.spacer: str = sequence.replace("U", "T")
        self.pam: PAM = pam
        self.position: int = position

    def bind(self, target: str, target_PAM: PAM, cutoff: float = 0.5) -> bool:
        """
        Predicts whether the gRNA will bind to a target

        :param target: The target
        :param target_PAM: The PAM of the target
        :param cutoff: The cutoff for the penalty between the gRNA and target
        :return: Binary value (True or False) of whether the gRNA will bing to the target
        """
        penalty: float = self.scorer.penalty(self.spacer, target, self.pam, target_PAM)
        return penalty < cutoff

    def binds(self, targets: list, target_PAMs: list, cutoff: float = 0.5) -> List[bool]:
        """
        Predicts whether the gRNA will bind to the specified targets

        :param targets: The list of targets
        :param target_PAMs: The corresponding list of PAMs
        :param cutoff: The cutoff for the penalty between the gRNA and target
        :return: Corresponding binary values (True or False) of whether the gRNA will bing to the targets
        """
        a = []
        for i in zip(targets, target_PAMs):
            a.append(self.bind(i[0], i[1], cutoff))
        return a

    def bind_mult(self, targets: list, target_PAMs: list, cutoff: float = 0.5) -> List[bool]:
        """
        Predicts whether the gRNA will bind to the specified targets using a Pool

        :param targets: The list of targets
        :param target_PAMs: The corresponding list of PAMs
        :param cutoff: The cutoff for the penalty between the gRNA and target
        :return: Corresponding binary values (True or False) of whether the gRNA will bing to the targets
        """
        out: List[bool] = gRNA._pool.starmap(self.bind, zip(targets, target_PAMs, [cutoff] * len(targets)))
        return out

    def generate_confusion_matrix(self, positives: list, positive_PAMs: list, negatives: list, negative_PAMs: list,
                                  cutoff: float = 0.5) -> ConfusionMatrix:
        """
        Generates a confusion matrix using the predicted binding ability between the gRNA and positive/negative sequences

        :param positives: Sequences that are intended to be targeted
        :param positive_PAMs: PAMs of sequences that are intended to be targeted
        :param negatives: Sequences that are not intended to be targeted
        :param negative_PAMs: PAMs of sequences that are not intended to be targeted
        :param cutoff: The cutoff for the penalty between the gRNA and sequences
        :return: Confusion matrix
        """
        tp = sum(self.bind_mult(positives, positive_PAMs, cutoff))
        fp = sum(self.bind_mult(negatives, negative_PAMs, cutoff))
        return {"tp": tp, "fn": len(positives) - tp, "tn": len(negatives) - fp, "fp": fp}

    @staticmethod
    def parallelize(cores: int = multiprocessing.cpu_count()):
        def parallelizable(method):
            def wrapper(*args, **kwargs):
                gRNA._pool = Pool(cores)
                try:
                    res = method(*args, **kwargs)
                    user_quit = False
                except BaseException:
                    traceback.print_exc()
                    user_quit = True
                gRNA._pool.close()
                gRNA._pool.join()

                if user_quit:
                    sys.exit(1)
                return res

            return wrapper

        return parallelizable

    def __eq__(self, other):
        """
        Returns a strict equality between two gRNA objects
        :param other:
        :return: bool
        """
        if not isinstance(other, gRNA):
            return False
        return self.spacer == other.spacer and self.position == other.position and self.scorer == other.scorer and self.pam == other.pam

    def spacer_eq(self, other):
        """
        Spacer equality between two gRNAs
        :param other: Other gRNA
        :return: bool
        """
        if isinstance(other, gRNA):
            return self.spacer == other.spacer
        if isinstance(other, str):
            return self.spacer == other
        return False

    def __add__(self, other):
        if type(other) == str:
            return self.__str__() + other
        else:
            return super().__add__(other)

    def __hash__(self):
        return hash(self.spacer)


class gRNA_Factory:
    def __init__(self, pam: PAM, length: int, scorer: Scorer):
        """
        Create gRNA_Factory to generate gRNAs of a certain length and having a certain PAM

        :param pam: The PAM for all generated gRNAs
        :param length: The length for all generated gRNAs
        :param scorer: The scorer for all generated gRNAs
        """
        self.pam = pam
        self.length = length
        self.scorer = scorer

    def create_gRNAs_with_reverse_complement(
            self,
            genomes_target: Union[str, List[str]],
            scoring_metric: Callable[[int, int, int, int], float],
            genomes_miss: Union[str, List[str]] = [],
            pam_minimum_prevalence: float = 0.5) -> pd.DataFrame:
        cur_targets = get_sequences_from_dir(genomes_target, ["fasta", "fna", "fastq"]) if type(
            genomes_target) == str else genomes_target
        cur_misses = get_sequences_from_dir(genomes_miss, ["fasta", "fna", "fastq"]) if type(
            genomes_miss) == str else genomes_miss
        print(f"{attr('bold')} Generating forward gRNAs {attr('reset')}")
        forward = self.create_gRNAs(
            cur_targets,
            scoring_metric,
            genomes_miss=cur_misses,
            pam_minimum_prevalence=pam_minimum_prevalence)
        cur_targets = ["".join([complement_table[nuc] for nuc in target[::-1]]) for target in cur_targets]
        cur_misses = ["".join([complement_table[nuc] for nuc in miss[::-1]]) for miss in
                      cur_misses]
        print(f"{attr('bold')} Generating reverse gRNAs {attr('reset')}")
        reverse = self.create_gRNAs(
            cur_targets,
            scoring_metric,
            genomes_miss=cur_misses,
            pam_minimum_prevalence=pam_minimum_prevalence)
        forward["direction"] = "forward"
        reverse["direction"] = "reverse"
        return pd.concat([forward, reverse])

    @gRNA.parallelize()
    def create_gRNAs(
            self,
            genomes_target: Union[str, List[str]],
            scoring_metric: Callable[[int, int, int, int], float],
            genomes_miss: Union[str, List[str]] = [],
            pam_minimum_prevalence: float = 0.5,
            bind_cutoff: float = 0.5) -> pd.DataFrame:
        """
        Create gRNAs sensitive to the sequences in 'genomes_dir'.

        :param bind_cutoff:
        :param genomes_target: directory for the aligned target sequences
        :param scoring_metric: The scoring metric (which takes a confusion matrix as input) to rank gRNAs by
        :param genomes_miss: directory for the aligned sequences not intended to be targeted
        :param pam_minimum_prevalence: The minimum prevalence of the PAM in the target sequences for a gRNA at that position to be considered
        """

        def wrapper_scoring_metric(series: pd.core.series.Series):
            """
            This is a wrapper that takes in a series and returns the score

            :param series: The pandas series containing "tp", "fn", "tn", and "fp" attributes
            :return: The evaluated scoring_metric
            """
            return scoring_metric(series.tp, series.fn, series.tn, series.fp)

        pams_lookup: Dict[str, PAM] = {str(pam): pam for pam in
                                       PAM(self.pam.location, self.pam.sequence).generate_non_ambiguous()}

        cur_targets: List[str] = get_sequences_from_dir(genomes_target, ["fasta", "fna", "fastq"]) if str == type(
            genomes_target) else list(genomes_target)
        """The positive target sequences"""
        number_targets: int = len(cur_targets)
        """Count of positive target sequences"""

        cur_misses: List[str] = get_sequences_from_dir(str(genomes_miss), ["fasta", "fna", "fastq"]) if str == type(
            genomes_miss) else list(genomes_miss)
        """The negative target sequences"""
        number_misses: int = len(cur_misses)
        """Count of negative target sequences"""

        nuc_prevalence: List[Counter] = [Counter(nucs) for nucs in zip(*cur_targets)]
        """The absolute nucleotide prevalence of the positive target sequences"""
        if self.pam.location == End(5):
            start_pam_search: int = 0
            """The position to start the PAM search"""
            end_pam_search: int = len(cur_targets[0]) - self.pam.length + 1 - self.length
            """The position to end the PAM search"""
            is_end_3: bool = False
            """Is the PAM at the 3' End?"""
        else:
            start_pam_search = self.length
            end_pam_search = len(cur_targets[0]) - self.pam.length + 1
            is_end_3 = True
        i: int = start_pam_search
        """Index of the current search region"""
        possible_pam_regions: List[int] = []
        """List of good candidate gRNA regions (have PAM prevalence above cutoff)"""
        while i < end_pam_search:
            # -----------------------------
            # Very quick filter to get rid of very bad PAM guesses...
            bad_pam_location: bool = False
            """Is this a bad PAM region?"""
            for i_pam, nuc_pam in enumerate(self.pam.sequence):
                count: int = 0
                """The count of the current nucleotide within this PAM region"""
                for seq_nuc in nuc_prevalence[i_pam + i].keys():
                    if PAM.overlap(PAM(self.pam.location, nuc_pam), PAM(self.pam.location, seq_nuc)):
                        count += nuc_prevalence[i_pam + i][seq_nuc]
                if count / number_targets < pam_minimum_prevalence:
                    bad_pam_location = True
                    break
            if bad_pam_location:
                i += 1
                continue
            # -----------------------------
            pams: List[str] = []
            "All the PAMs at this position within all positive target sequences"
            for target in cur_targets:
                pams.append(target[i: i + self.pam.length])
            count_good: int = 0
            """Count of good PAMs [that is suitable for the gRNA]"""

            for pam, prev in Counter(pams).items():
                if PAM.overlap(PAM(End(self.pam.location), pam), self.pam):
                    count_good += prev
            if count_good / number_targets < pam_minimum_prevalence:
                i += 1
                continue
            possible_pam_regions.append(i)
            i += 1
        # df_pre_type_unit = {"location": int, "gRNA": gRNA, "binding_efficiency": float}
        df_pre: List[dict] = []
        pam_place: int
        for pam_place in tqdm(possible_pam_regions, desc="gRNAs at this position"):
            spacers_list: List[str] = []
            targets_w_pam: List[str] = []
            for target in cur_targets:
                if is_end_3:
                    spacers_list.append(target[pam_place - self.length:pam_place])
                    targets_w_pam.append(target[pam_place - self.length:pam_place + self.pam.length])
                else:
                    spacers_list.append(target[pam_place + self.pam.length:pam_place + self.pam.length + self.length])
                    targets_w_pam.append(target[pam_place:pam_place + self.pam.length + self.length])
            misses_w_pam: List[str] = []
            for target in cur_misses:
                if is_end_3:
                    misses_w_pam.append(target[pam_place - self.length:pam_place + self.pam.length])
                else:
                    misses_w_pam.append(target[pam_place:pam_place + self.pam.length + self.length])
            mwp_count: Counter = Counter(misses_w_pam)

            spacers: List[str] = list(Counter(spacers_list).keys())
            twp_count: Counter = Counter(targets_w_pam)
            for spacer in spacers:
                if is_end_3:
                    PAMs, targets = zip(
                        *[(helper_get_PAM(pams_lookup, twp[-self.pam.length:], self.pam.location),
                           twp[:self.length]) for twp in twp_count.keys()])
                    PAMs_misses, misses = zip(
                        *[(helper_get_PAM(pams_lookup, mwp[-self.pam.length:], self.pam.location), mwp[:self.length]) for mwp in
                            mwp_count.keys()]) if genomes_miss else ([], [])
                else:
                    PAMs, targets = zip(
                        *[(helper_get_PAM(pams_lookup, twp[:self.pam.length], self.pam.location),
                            twp[self.pam.length:self.pam.length + self.length]) for
                          twp in twp_count.keys()])
                    PAMs_misses, misses = zip(
                        *[(helper_get_PAM(pams_lookup, mwp[:self.pam.length], self.pam.location),
                            mwp[self.pam.length:self.pam.length + self.length]) for
                          mwp in mwp_count.keys()]) if genomes_miss else ([], [])

                this_gRNA = gRNA(pam_place + self.pam.length if not is_end_3 else pam_place - self.length,
                                 spacer.replace("T", "U"), self.pam, self.scorer)
                binding_results_positive = this_gRNA.binds(targets, PAMs, cutoff=bind_cutoff)
                binding_results_negative = this_gRNA.binds(misses, PAMs_misses, cutoff=bind_cutoff)

                binding_efficiency_positive: float = 0
                for target_w_pam, binding_result in zip(twp_count.keys(), binding_results_positive):
                    binding_efficiency_positive += binding_result * twp_count[target_w_pam]
                binding_efficiency_negative: float = 0
                for miss_w_pam, binding_result in zip(mwp_count.keys(), binding_results_negative):
                    binding_efficiency_negative += binding_result * mwp_count[miss_w_pam]
                df_pre.append({
                    "location": pam_place + self.pam.length if not is_end_3 else pam_place - self.length,
                    "gRNA": this_gRNA,
                    "tp": binding_efficiency_positive,
                    "fn": number_targets - binding_efficiency_positive,
                    "tn": number_misses - binding_efficiency_negative,
                    "fp": binding_efficiency_negative,
                })
        df = pd.DataFrame(df_pre)
        if df.shape[0] != 0:
            df[scoring_metric.__name__] = df.apply(wrapper_scoring_metric, axis=1)
            return df.sort_values(by=scoring_metric.__name__, ascending=False).reset_index(drop=True)
        else:
            return df


def helper_get_PAM(pam_dict: Dict, query_str: str, pam_end):
    res = pam_dict.get(query_str, -1)
    if res == -1:
        res = PAM(pam_end, query_str)
        pam_dict[query_str] = res
    return res
