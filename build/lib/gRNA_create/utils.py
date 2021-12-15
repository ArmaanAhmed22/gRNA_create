import os
from typing import Callable, TypedDict, List, Dict

from Bio import SeqIO


def get_sequences_from_dir(dir: str, accepted_extensions: List[str]) -> List[str]:
    """
    This utility function gets sequences from a directory

    :param dir: The directory to get the sequences from
    :param accepted_extensions: The files that have one of these extensions will be read
    :return: List of the sequences
    """
    if ("." in dir):
        return [str(record.seq) for record in list(SeqIO.parse(open(dir), dir.split(".")[-1]))]
    genomes: List[str] = return_files_with_extension(dir, accepted_extensions)
    cur_targets: List[str] = []
    for g in genomes:
        cur_targets += [str(record.seq) for record in list(SeqIO.parse(open(dir + "/" + g[0]), g[1]))]
    return cur_targets


def return_files_with_extension(files_dir: str, extensions: List[str]) -> list:
    """
    This utility function returns files from a directory containing of the extensions

    :param files_dir: The directory to get the files from
    :param extensions: The extensions which the returned files need to have one of
    :return: File names (including extension) with one of the specified extensions
    """
    listing = os.listdir(files_dir)
    cur_files = [(cur_listing, ext) for cur_listing in listing if (ext := cur_listing.split(".")[-1]) in extensions]
    return cur_files


complement_table: Dict[str, str] = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G"}

transcribe_table: Dict[str, str] = {
    "A": "A",
    "T": "U",
    "G": "G",
    "C": "C"
}

dna_letters = "ACGT"
rna_letters = "ACGU"

nucleotide_code_bidirectional_dna = {2 ** i if n else nuc: nuc if n else 2 ** i for i, nuc in enumerate(dna_letters) for
                                     n in range(2)}
nucleotide_code_bidirectional_rna = {2 ** i if n else nuc: nuc if n else 2 ** i for i, nuc in enumerate(rna_letters) for
                                     n in range(2)}

nucleotide_index_dna = {i if n else nuc: nuc if n else i for i, nuc in enumerate("ACGT") for n in range(2)}
nucleotide_index_rna = {i if n else nuc: nuc if n else i for i, nuc in enumerate("ACGU") for n in range(2)}


def reverse_complement_dna(seq):
    return "".join([complement_table[nuc] for nuc in seq[::-1]])


ambiguous_code: Dict[str, int] = {
    "A": nucleotide_code_bidirectional_dna["A"],
    "C": nucleotide_code_bidirectional_dna["C"],
    "G": nucleotide_code_bidirectional_dna["G"],
    "T": nucleotide_code_bidirectional_dna["T"],
    "M":
        nucleotide_code_bidirectional_dna["A"] +
        nucleotide_code_bidirectional_dna["C"],
    "R":
        nucleotide_code_bidirectional_dna["A"] +
        nucleotide_code_bidirectional_dna["G"],
    "W":
        nucleotide_code_bidirectional_dna["A"] +
        nucleotide_code_bidirectional_dna["T"],
    "S":
        nucleotide_code_bidirectional_dna["C"] +
        nucleotide_code_bidirectional_dna["G"],
    "Y":
        nucleotide_code_bidirectional_dna["C"] +
        nucleotide_code_bidirectional_dna["T"],
    "K":
        nucleotide_code_bidirectional_dna["G"] +
        nucleotide_code_bidirectional_dna["T"],
    "V":
        nucleotide_code_bidirectional_dna["A"] +
        nucleotide_code_bidirectional_dna["C"] +
        nucleotide_code_bidirectional_dna["G"],
    "H":
        nucleotide_code_bidirectional_dna["A"] +
        nucleotide_code_bidirectional_dna["C"] +
        nucleotide_code_bidirectional_dna["T"],
    "D":
        nucleotide_code_bidirectional_dna["A"] +
        nucleotide_code_bidirectional_dna["G"] +
        nucleotide_code_bidirectional_dna["T"],
    "B":
        nucleotide_code_bidirectional_dna["C"] +
        nucleotide_code_bidirectional_dna["G"] +
        nucleotide_code_bidirectional_dna["T"],
    "N":
        nucleotide_code_bidirectional_dna["A"] +
        nucleotide_code_bidirectional_dna["C"] +
        nucleotide_code_bidirectional_dna["G"] +
        nucleotide_code_bidirectional_dna["T"]
}


def sensitivity(tp: int, fn: int, tn: int, fp: int) -> float:
    return tp / (tp + fn)


def specificity(tp: int, fn: int, tn: int, fp: int) -> float:
    return tn / (tn + fp)


def precision(tp: int, fn: int, tn: int, fp: int) -> float:
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def f_score(theta: float) -> Callable[[int, int, int, int], float]:
    """
    This function returns a specified f-score.
    For example:
        f_score( theta = 1 ) returns f_score_1
        f_score( theta = 2 ) returns f_score_2

    :param theta: The theta value for the specific f-score
    :return: The specified f-score function
    """

    def f(tp: int, fn: int, tn: int, fp: int) -> float:
        p: float = precision(tp, fn, tn, fp)
        s: float = sensitivity(tp, fn, tn, fp)
        if s == 0 or p == 0:
            return 0
        return (1 + theta ** 2) * p * s / (theta ** 2 * p + s)

    f.__name__ = f"f_score_{theta}"
    return f


class ConfusionMatrix(TypedDict):
    tp: int
    fn: int
    tn: int
    fp: int
