from __future__ import annotations
from collections import Counter

from typing import Callable, List, cast, Dict

import pandas
import pygad
import seaborn

from gRNA_create.gRNA import gRNA
from gRNA_create.gRNA_scorer import Scorer
from gRNA_create.pam import PAM
from gRNA_create.utils import nucleotide_index_rna, ConfusionMatrix, \
    reverse_complement_dna
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

gRNA_alias = List[int]


class gRNAGeneticAlgorithm(pygad.GA):

    def __init__(self,
                 num_generations: int,
                 num_parents_mating: int,
                 scoring_metric: Callable[[ConfusionMatrix], float],
                 initial_gRNAs: List[gRNA],
                 targets: List[str],
                 misses: List[str] = [],
                 mutation_rate: float = 0.1,
                 binding_cutoff: float = 0.5,
                 **kwargs
                 ):
        """gRNA genetic algorithm for a signel gRNA location
        Args:
            num_generations (int): Number of generations to run algorithm for
            num_parents_mating (int): Number of gRNAs to splice together during "mating"
            scoring_metric (Callable[[ConfusionMatrix], float]): The scoring metric
            initial_gRNAs (List[gRNA]): The initial population of gRNAs
            targets (List[str]): The aligned target sequences
            misses (List[str], optional): The aligned negative sequences. Defaults to [].
            mutation_rate (float, optional): The chance of changing any nucleotide in any gRNA. Defaults to 0.1.
            binding_cutoff (float, optional): The cutoff score for binding. Defaults to 0.5.
        """
        assert len(set([g.position for g in initial_gRNAs])) == 1
        assert len(set([g.pam for g in initial_gRNAs])) == 1
        assert len(set([g.scorer for g in initial_gRNAs])) == 1
        assert len(set([len(g) for g in initial_gRNAs])) == 1

        gRNA_aliases = [tuple([nucleotide_index_rna[nuc] for nuc in cur_gRNA]) for cur_gRNA in initial_gRNAs]

        self.pos = initial_gRNAs[0].position
        self.pam = initial_gRNAs[0].pam
        self.scorer = initial_gRNAs[0].scorer
        self.length = len(initial_gRNAs[0])

        self.scoring_metric = scoring_metric

        self.target_length = len(targets)
        self.miss_length = len(misses)

        self.binding_cutoff = binding_cutoff

        short_targets = [target[self.pos:self.pos + self.length] for target in targets]
        target_PAMs = [
            target[self.pos - self.pam.length: self.pos]
            if self.pam.location.value == 5
            else target[self.pos + self.length: self.pos + self.length + self.pam.length]
            for target in targets]
        self.PAM_and_targets_count = Counter(list(zip(target_PAMs, short_targets)))
        short_misses = [miss[self.pos:self.pos + self.length] for miss in misses]
        miss_PAMs = [miss[self.pos - self.pam.length: self.pos]
                     if self.pam.location.value == 5
                     else miss[self.pos + self.length: self.pos + self.length + self.pam.length]
                     for miss in misses] if misses is not None else []

        self.PAM_and_misses_count = Counter(list(zip(miss_PAMs, short_misses)))
        self.generation = 1

        def on_generation(x):
            if self.generation > 1:
                end: str = ""
                if self.generation == num_generations:
                    end = "\n"
                print("\b" * (len(str(self.generation - 1)) + 3 + len(str(num_generations))), end=end)
            else:
                print()
            print(self.generation, "/", num_generations, end="")
            self.generation += 1

        super().__init__(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            gene_type=int,
            gene_space=range(4),
            initial_population=np.array(gRNA_aliases),
            fitness_func=_return_fitness_func(self.pos,
                                              self.pam,
                                              self.scorer,
                                              self.PAM_and_targets_count,
                                              self.PAM_and_misses_count,
                                              self.target_length,
                                              self.miss_length,
                                              self.scoring_metric,
                                              self.binding_cutoff),
            on_generation=on_generation,
            mutation_probability=mutation_rate,
            **kwargs
        )


def _return_fitness_func(pos: int, pam: PAM, scorer: Scorer, PAM_and_targets_count, PAM_and_misses_count, target_length,
                         miss_length, scoring_metric, binding_cutoff: float):
    def fitness_func(chromosome: gRNA_alias, idx: int):

        gRNA_str: str = "".join([str(nucleotide_index_rna[nuc_index]) for nuc_index in chromosome])

        cur_gRNA: gRNA = gRNA(pos, gRNA_str, pam, scorer)

        PAM_and_targets = list(PAM_and_targets_count.keys())
        PAM_and_misses = list(PAM_and_misses_count.keys())

        target_PAMs, targets = zip(*PAM_and_targets)
        miss_PAMs, misses = zip(*PAM_and_misses) if miss_length > 0 else ([], [])

        target_PAMs = [PAM(pam.location, p) for p in target_PAMs]
        miss_PAMs = [PAM(pam.location, p) for p in miss_PAMs]

        confusion_matrix: ConfusionMatrix = cast(ConfusionMatrix, {p: 0 for p in ["tp", "fn", "tn", "fp"]})
        for i, res in enumerate(cur_gRNA.binds(targets, target_PAMs, cutoff=binding_cutoff)):
            confusion_matrix["tp"] += res * PAM_and_targets_count[PAM_and_targets[i]]
        confusion_matrix["fn"] = target_length - confusion_matrix["tp"]
        if miss_length > 0:
            for i, res in enumerate(cur_gRNA.binds(misses, miss_PAMs, cutoff=binding_cutoff)):
                confusion_matrix["fp"] += res * PAM_and_misses_count[PAM_and_misses[i]]
            confusion_matrix["tn"] = miss_length - confusion_matrix["fp"]

        return scoring_metric(**confusion_matrix)

    return fitness_func


class gRNAGroupGA:

    def __init__(self,
                 num_generations: int,
                 num_parents_mating: int,
                 scoring_metric: Callable[[ConfusionMatrix], float],
                 initial_gRNAs: pandas.DataFrame,
                 targets: List[str],
                 misses: List[str] = [],
                 mutation_rate: float = 0.1,
                 multiplier: int = 1,
                 **kwargs
                 ):
        """gRNA genetic algorithm for a whole population of gRNAs

        Args:
            num_generations (int): The number of generations to run the genetic algorithm for.
            num_parents_mating (int): The number of gRNAs to splice together.
            scoring_metric (Callable[[ConfusionMatrix], float]): The scoring metric to measure gRNA performance.
            initial_gRNAs (pandas.DataFrame): Initial population of gRNAs.
            targets (List[str]): List of aligned target regions.
            misses (List[str], optional): List of aligned negative regions. Defaults to [].
            mutation_rate (float, optional): The chance of inserting a mutation into any given position of any gRNA. Defaults to 0.1.
            multiplier (int, optional): The scalar to multiply the population of gRNAs by. Useful if there are not enough parents. Defaults to 1.
        """
        self.pos_to_gRNAs: Dict[int, Dict[str, gRNA]] = {}
        self.scoring_metric: Callable[[ConfusionMatrix], float] = scoring_metric
        for _, cur_row in initial_gRNAs.iterrows():
            for _ in range(multiplier):
                if self.pos_to_gRNAs.get(cur_row['gRNA'].position, -1) == -1:
                    self.pos_to_gRNAs[cur_row['gRNA'].position] = cast(Dict[str, gRNA], {"forward": [], "reverse": []})
                if "direction" in cur_row and cur_row["direction"] == "reverse":
                    self.pos_to_gRNAs[cur_row['gRNA'].position]["reverse"].append(cur_row["gRNA"])
                else:
                    self.pos_to_gRNAs[cur_row['gRNA'].position]["forward"].append(cur_row["gRNA"])

        self.gRNA_genetic_algorithms: List[gRNAGeneticAlgorithm] = []
        for pos, g_dicts in self.pos_to_gRNAs.items():
            for dir, cur_gRNAs in g_dicts.items():
                if dir == "reverse" and len(cur_gRNAs) != 0:
                    self.gRNA_genetic_algorithms.append(
                        gRNAGeneticAlgorithm(num_generations, num_parents_mating, scoring_metric, cur_gRNAs,
                                             [reverse_complement_dna(cur_target) for cur_target in targets],
                                             [reverse_complement_dna(cur_miss) for cur_miss in misses],
                                             mutation_rate=mutation_rate, **kwargs))
                elif dir == "forward" and len(cur_gRNAs) != 0:
                    self.gRNA_genetic_algorithms.append(
                        gRNAGeneticAlgorithm(num_generations, num_parents_mating, scoring_metric, cur_gRNAs, targets,
                                             misses, mutation_rate=mutation_rate, **kwargs))

    @gRNA.parallelize()
    def run(self, show_end_graph: bool = False):
        """Run the genetic algorithm

        Args:
            show_end_graph (bool, optional): Whether an end graph should be shown to see gRNA improvements. Defaults to False.
        """
        df_pre = []

        for i, GA in tqdm(enumerate(self.gRNA_genetic_algorithms), total=len(self.gRNA_genetic_algorithms)):
            GA.run()
            df_pre += [{"pos": list(self.pos_to_gRNAs.keys())[i], "best": sol, "generation": gen_num} for gen_num, sol
                       in enumerate(GA.best_solutions_fitness)]
        if show_end_graph:
            seaborn.lineplot(x="generation", y="best", data=pandas.DataFrame(df_pre), hue="pos")
            plt.show()

    def get_result(self) -> pandas.DataFrame:
        """Return genetic algorithm results

        Returns:
            pandas.DataFrame: Dataframe of new population of improved gRNAs.
        """
        gRNA_to_metric_score: dict = {}
        for i, pos in enumerate(self.pos_to_gRNAs.keys()):
            cur_GA: gRNAGeneticAlgorithm = self.gRNA_genetic_algorithms[i]
            for chromosome, fitness in zip(cur_GA.population, cur_GA.last_generation_fitness):
                cur_gRNA: gRNA = gRNA(pos, "".join([str(nucleotide_index_rna[nuc]) for nuc in chromosome]), cur_GA.pam,
                                      cur_GA.scorer)
                gRNA_to_metric_score[cur_gRNA] = fitness

        df_pre: List[dict] = \
            [{"location": cur_gRNA.position, "gRNA": cur_gRNA, self.scoring_metric.__name__: score} for cur_gRNA, score
             in gRNA_to_metric_score.items()]
        return pandas.DataFrame(df_pre)
