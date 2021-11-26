from typing import List

from Bio import SeqIO

positive_seqs: List[str] = \
    [str(record.seq) for record in SeqIO.parse(open("tests/test_sequences.fasta"), format="fasta")]

negative_seqs: List[str] = \
    [str(record.seq) for record in SeqIO.parse(open("tests/test_sequences_neg.fasta"), format="fasta")]

positive_seqs_large: List[str] = \
    [str(record.seq) for record in SeqIO.parse(open("tests/test_sequences_large.fasta"), format="fasta")]
