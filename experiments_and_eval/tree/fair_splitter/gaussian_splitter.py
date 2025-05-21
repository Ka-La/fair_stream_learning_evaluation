from __future__ import annotations

import math

from river.base.typing import ClfTarget
from river.proba import Gaussian

from river.tree.utils import BranchFactory
from .base import Splitter


class GaussianSplitter(Splitter):
    """Numeric attribute observer for classification tasks that is based on
    Gaussian estimators.

    The distribution of each class is approximated using a Gaussian distribution. Hence,
    the probability density function can be easily calculated.


    Parameters
    ----------
    n_splits
        The number of partitions to consider when querying for split candidates.
    """

    def __init__(self, deprivedIndex, granted=1.0, n_splits: int = 10):
        super().__init__()
        self._min_per_class: dict[ClfTarget, float] = {}
        self._max_per_class: dict[ClfTarget, float] = {}
        self._att_dist_per_class: dict[ClfTarget, Gaussian] = {}

        self.deprivedIndex = deprivedIndex
        self.granted = granted
        self.dTD = 'deprivedTotalDist'
        self.unTD = 'undeprivedTotalDist'
        self.dGD = 'deprivedGrantedDist'
        self.unGD = 'undeprivedGrantedDist'

        #initialize min and max values for each discrimination related normal distribution
        self._min_disc_per_gauss = {}
        self._min_disc_per_gauss[self.dTD] = math.inf
        self._min_disc_per_gauss[self.unTD] = math.inf
        self._min_disc_per_gauss[self.dGD] = math.inf
        self._min_disc_per_gauss[self.unGD] = math.inf

        self._max_disc_per_gauss = {}
        self._max_disc_per_gauss[self.dTD] = - math.inf
        self._max_disc_per_gauss[self.unTD] = - math.inf
        self._max_disc_per_gauss[self.dGD] = - math.inf
        self._max_disc_per_gauss[self.unGD] = - math.inf

        #immediately initialize Gaussian's for all discrimination cases
        self._disc_dist_per_group: dict[type.String, Gaussian] = {}
        self._disc_dist_per_group[self.dTD] = Gaussian()
        self._disc_dist_per_group[self.unTD] = Gaussian()
        self._disc_dist_per_group[self.dGD] = Gaussian()
        self._disc_dist_per_group[self.unGD] = Gaussian()

        self.n_splits = n_splits

    
    #helper function to update discrimination
    def update_disc_gaussian(self, att_val, w, category):
        if(att_val < self._min_disc_per_gauss[category]):
            self._min_disc_per_gauss[category] = att_val

        if(att_val > self._max_disc_per_gauss[category]):
            self._max_disc_per_gauss[category] = att_val

        self._disc_dist_per_group[category].update(att_val, w)
        return

    #this also needs the sensitive attribute value
    def update(self, att_val, target_val, w, sen_att_val):
        if att_val is None:
            return
        else:
            try:
                val_dist = self._att_dist_per_class[target_val]
                if att_val < self._min_per_class[target_val]:
                    self._min_per_class[target_val] = att_val
                if att_val > self._max_per_class[target_val]:
                    self._max_per_class[target_val] = att_val
            except KeyError:
                val_dist = Gaussian()
                self._att_dist_per_class[target_val] = val_dist
                self._min_per_class[target_val] = att_val
                self._max_per_class[target_val] = att_val

            val_dist.update(att_val, w)

            if (sen_att_val == self.deprivedIndex):
                #update deprivedTotalDist
                self.update_disc_gaussian(att_val, w, self.dTD)

                if(float(target_val) == self.granted):
                    #update deprivedGrantedDist
                    self.update_disc_gaussian(att_val, w, self.dGD)

            else:
                #update undeprivedTotalDist
                self.update_disc_gaussian(att_val, w, self.unTD)

                if(float(target_val) == self.granted):
                    #update undeprivedGrantedDist
                    self.update_disc_gaussian(att_val, w, self.unGD)




        

    def cond_proba(self, att_val, target_val):
        if target_val in self._att_dist_per_class:
            obs = self._att_dist_per_class[target_val]
            return obs(att_val)
        else:
            return 0.0
        
    def split_val_discrimination(self, split_value, normalize=False, weigh=False):

        #left part of the split_val
        leftDist = {}
        leftDist[self.dTD] = None
        leftDist[self.unTD] = None
        leftDist[self.dGD] = None
        leftDist[self.unGD] = None

        #right part of the split_val    
        rightDist = {}
        rightDist[self.dTD] = None
        rightDist[self.unTD] = None
        rightDist[self.dGD] = None
        rightDist[self.unGD] = None

        #obtain related discrimination counts for left and right splits

        for disc_val, disc_est in self._disc_dist_per_group.items():

            if (disc_est != None):
                if (split_value < self._min_disc_per_gauss[disc_val]):
                    rightDist[disc_val] = disc_est.n_samples

                elif (split_value > self._max_disc_per_gauss[disc_val]):
                    leftDist[disc_val] = disc_est.n_samples
                else:
                    leftDist[disc_val] = disc_est.cdf(split_value) + disc_est.n_samples
                    rightDist[disc_val] = disc_est.n_samples - leftDist[disc_val]

        #Statistical Parity Calculation based on the split
               
        #left side
                    
        leftUnTotalCount = leftUnGrantedCount = leftDTotalCount = leftDGrantedCount = 0
        leftUnRate = leftDRate = leftDisc = 0

        if (leftDist[self.dTD] != None):
            leftDTotalCount = leftDist[self.dTD]   
        if (leftDist[self.unTD] != None):
            leftUnTotalCount = leftDist[self.unTD]   
        if (leftDist[self.dGD] != None):
            leftDGrantedCount = leftDist[self.dGD]   
        if (leftDist[self.unGD] != None):
            leftUnGrantedCount = leftDist[self.unGD]
        
        if (leftUnTotalCount != 0):
            leftUnRate = leftUnGrantedCount / leftUnTotalCount

        if (leftDTotalCount != 0):
            leftDRate = leftDGrantedCount / leftDTotalCount
        
        leftDisc = abs(leftUnRate - leftDRate)

        #right side

        rightUnTotalCount = rightUnGrantedCount = rightDTotalCount = rightDGrantedCount = 0
        rightUnRate = rightDRate = rightDisc = 0

        if (rightDist[self.dTD] != None):
            rightDTotalCount = rightDist[self.dTD]   
        if (rightDist[self.unTD] != None):
            rightUnTotalCount = rightDist[self.unTD]   
        if (rightDist[self.dGD] != None):
            rightDGrantedCount = rightDist[self.dGD]   
        if (rightDist[self.unGD] != None):
            rightUnGrantedCount = rightDist[self.unGD]
        
        if (rightUnTotalCount != 0):
            rightUnRate = rightUnGrantedCount / rightUnTotalCount

        if (rightDTotalCount != 0):
            rightDRate = rightDGrantedCount / rightDTotalCount
        
        rightDisc = abs(rightUnRate - rightDRate)

        #sum of left and right gets returned (might need to be weighed by number of samples)

        if weigh: #if weighed, we weigh the disc sides
            right_samples = rightDTotalCount+rightUnTotalCount
            left_samples = leftDTotalCount + leftUnTotalCount

            right_weight = right_samples/(right_samples+left_samples)
            left_weight = left_samples/(left_samples+right_samples)

            disc = abs(left_weight*leftDisc + right_weight*rightDisc)
        else:
            disc = abs(leftDisc + rightDisc) #normalizing because it makes sense
            if normalize: #if normalize, we normalize the disc by dividing by 2 for the number of sides
                disc = float(disc/2)

        return disc

    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only, node, sens_att_name, trade_off=1):
        best_suggestion = BranchFactory()

        suggested_split_values = self._split_point_suggestions()
        for split_value in suggested_split_values:
            post_split_dist = self._class_dists_from_binary_split(split_value)
            post_disc_merit = self.split_val_discrimination(split_value,weigh=criterion.weigh_discrimination , normalize=criterion.normalize_discrimination)

            merit = criterion.merit_of_split(pre_split_dist, post_split_dist, att_idx, post_disc_merit, node, sens_att_name, trade_off)
            if merit > best_suggestion.merit:
                best_suggestion = BranchFactory(merit, att_idx, split_value, post_split_dist)

        return best_suggestion
    

    def _split_point_suggestions(self):
        suggested_split_values = []
        min_value = math.inf
        max_value = -math.inf
        for k, estimator in self._att_dist_per_class.items():
            if self._min_per_class[k] < min_value:
                min_value = self._min_per_class[k]
            if self._max_per_class[k] > max_value:
                max_value = self._max_per_class[k]
        if min_value < math.inf:
            bin_size = max_value - min_value
            bin_size /= self.n_splits + 1.0
            for i in range(self.n_splits):
                split_value = min_value + (bin_size * (i + 1))
                if min_value < split_value < max_value:
                    suggested_split_values.append(split_value)
        return suggested_split_values

    def _class_dists_from_binary_split(self, split_value):
        lhs_dist = {}
        rhs_dist = {}
        for k, estimator in self._att_dist_per_class.items():
            if split_value < self._min_per_class[k]:
                rhs_dist[k] = estimator.n_samples
            elif split_value >= self._max_per_class[k]:
                lhs_dist[k] = estimator.n_samples
            else:
                lhs_dist[k] = estimator.cdf(split_value) * estimator.n_samples
                rhs_dist[k] = estimator.n_samples - lhs_dist[k]
        return [lhs_dist, rhs_dist]