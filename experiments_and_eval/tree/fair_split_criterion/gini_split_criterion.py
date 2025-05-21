from __future__ import annotations

import math

from .base import SplitCriterion

#TODO Make sure that it makes sense to multiply gini with disc loss as well (FAHT code only does for info gain)
class GiniSplitCriterion(SplitCriterion):
    """Gini Impurity split criterion."""

    def __init__(self, min_branch_fraction):
        super().__init__()
        self.min_branch_fraction = min_branch_fraction

    def merit_of_split(self, pre_split_dist, post_split_dist, attr_name, post_disc_merit, node, sens_att_name, trade_off):
        if self.num_subsets_greater_than_frac(post_split_dist, self.min_branch_fraction) < 2:
            return -math.inf

        total_weight = 0.0
        dist_weights = [0.0] * len(post_split_dist)
        for i in range(len(post_split_dist)):
            dist_weights[i] = sum(post_split_dist[i].values())
            total_weight += dist_weights[i]
        gini = 0.0
        for i in range(len(post_split_dist)):
            gini += (dist_weights[i] / total_weight) * self.compute_gini(
                post_split_dist[i], dist_weights[i]
            )
        gini_split =  1.0 - gini

        pre_disc = post_disc = disc_loss = 0.0

        sens_att_stats = node.splitters[sens_att_name]
        
        pre_disc = abs(sens_att_stats.calc_disc_per_att())

        if (post_disc_merit == -1):
            post_disc = pre_disc
        else:
            post_disc = abs(post_disc_merit)

        disc_loss = pre_disc - post_disc
        if disc_loss > 0.0:

            fair_gini = gini_split * disc_loss
        else:
            fair_gini = gini_split

        return fair_gini



    @staticmethod
    def compute_gini(dist, dist_sum_of_weights):
        gini = 1.0
        if dist_sum_of_weights != 0.0:
            for _, val in dist.items():
                rel_freq = val / dist_sum_of_weights
                gini -= rel_freq * rel_freq
        return gini

    @staticmethod
    def range_of_merit(pre_split_dist):
        return 1.0

    @staticmethod
    def num_subsets_greater_than_frac(distributions, min_frac):
        total_weight = 0.0
        dist_sums = [0.0] * len(distributions)
        for i in range(len(dist_sums)):
            dist_sums[i] = sum(distributions[i].values())
            total_weight += dist_sums[i]
        num_greater = 0

        if total_weight > 0:
            for d in dist_sums:
                if (d / total_weight) > min_frac:
                    num_greater += 1
        return num_greater