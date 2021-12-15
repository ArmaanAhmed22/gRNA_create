import math
from typing import Callable, List, Dict

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

from gRNA_create.gRNA import gRNA

from colored import attr


class gRNAVisualize:
    def __init__(self, data: pd.DataFrame, scoring_metric: Callable[[int, int, int, int], float]):
        self.data: pd.DataFrame = data
        self.scoring_metric_name: str = scoring_metric.__name__
        self.has_rev_comp: bool = "direction" in data.columns

    def create_rank_plot(self, output: str = "-", separate_rev_comp: bool = True, top_quantile: float = 0.8):
        sorted: pd.DataFrame = self.data.sort_values(by=self.scoring_metric_name, ascending=False)
        if self.has_rev_comp and separate_rev_comp:
            direction_breakup_names: List[str] = sorted["direction"].unique()
            direction_breakups: List[pd.DataFrame] = []
            for cur_name in direction_breakup_names:
                cur_df: pd.DataFrame = sorted[sorted["direction"] == cur_name]
                quant: float = cur_df[self.scoring_metric_name].quantile(q=top_quantile)
                direction_breakups.append(cur_df[cur_df[self.scoring_metric_name] >= quant].reset_index(drop=True))
            visualize_data: pd.DataFrame = pd.concat(direction_breakups)
            visualize_kwargs: Dict = {
                "hue": "direction"
            }
        else:
            visualize_data_quant: float = sorted[self.scoring_metric_name].quantile(q=top_quantile)
            visualize_data = sorted[sorted[self.scoring_metric_name] >= visualize_data_quant].reset_index(drop=True)
            visualize_kwargs = {}
        visualize_data["ranking"] = visualize_data.index

        fig, ax = plt.subplots()
        sns.lineplot(y=self.scoring_metric_name, x="ranking", data=visualize_data, ax=ax, **visualize_kwargs)
        sns.despine(ax=ax)
        change: int = 10**(math.floor(math.log10(len(visualize_data["ranking"].to_numpy()))))
        num_ticks: int = len(visualize_data["ranking"].to_numpy()) // change
        plt.xticks([change * i for i in range(num_ticks)])
        if output == "-":
            plt.show()
        else:
            plt.savefig(output, dpi=600, bbox_inches="tight")

    def compare_best_with(self, other_gRNAVisualizer):
        other_data: pd.DataFrame = other_gRNAVisualizer.data.sort_values(by=other_gRNAVisualizer.scoring_metric_name, ascending=False)
        best_gRNA: gRNA = other_data.gRNA.loc[0]
        best_loc: int = other_data.location.loc[0]

        cur_gRNA: gRNA = self.data[self.data["location"] == best_loc].sort_values(by=self.scoring_metric_name, ascending=False).gRNA.loc[0]
        str_out_best: str = ""
        for nuc_b, nuc_c in zip(best_gRNA, cur_gRNA):
            if nuc_b == nuc_c:
                str_out_best += nuc_b
            else:
                str_out_best += attr("bold") + nuc_b + attr("reset")
        print(cur_gRNA + "\n" + str_out_best)
