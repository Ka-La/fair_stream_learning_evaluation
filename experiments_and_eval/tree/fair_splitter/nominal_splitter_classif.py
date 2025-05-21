from __future__ import annotations

import collections

from river.tree.utils import BranchFactory
from .base import Splitter





class NominalSplitterClassif(Splitter):
    """Splitter utilized to monitor nominal features in classification tasks.

    As the monitored feature is nominal, it already has well-defined partitions. Hence,
    this splitter uses dictionary structures to keep class counts for each incoming category.
    """

    #default for granted is 1
    def __init__(self, deprivedIndex, granted_target_val=1.0):
        super().__init__()
        self._total_weight_observed = 0.0
        self._missing_weight_observed = 0.0
        self._att_dist_per_class = collections.defaultdict(dict)
        self._att_values = set()

        self.deprivedIndex = deprivedIndex
        self.granted = granted_target_val

        self.dTD = 'deprivedTotalDist'
        self.unTD = 'undeprivedTotalDist'
        self.dGD = 'deprivedGrantedDist'
        self.unGD = 'undeprivedGrantedDist'

        self._disc_dist_per_group = collections.defaultdict(dict)

    @property
    def is_numeric(self):
        return False
    
    def update_disc_dist(self, att_val, category, w):

        if (self._disc_dist_per_group[att_val][category] == None):
            self._disc_dist_per_group[att_val][category] = 1.0
        else:
            self._disc_dist_per_group[att_val][category] += w

        return
    
    def add_disc_dist(self,att_val, sens_att_val, class_val, w):

        #initialize the dictionary for the discrimination per att_val

        if not att_val in self._disc_dist_per_group.keys():
            self._disc_dist_per_group[att_val][self.dTD] = None
            self._disc_dist_per_group[att_val][self.unTD] = None
            self._disc_dist_per_group[att_val][self.dGD] = None
            self._disc_dist_per_group[att_val][self.unGD] = None


        if (sens_att_val == self.deprivedIndex):
            self.update_disc_dist(att_val, self.dTD, w)

            if (float(class_val) == self.granted):
                self.update_disc_dist(att_val, self.dGD, w)
        else:
            self.update_disc_dist(att_val, self.unTD, w)
            if (float(class_val) == self.granted):
                self.update_disc_dist(att_val, self.unGD, w)

        return
    
    def calc_disc_per_val(self, att_val):
        unTotalCount = unGrantedCount = dTotalCount = dGrantedCount = 0
        unRate = dRate = 0

        if(self._disc_dist_per_group[att_val][self.dTD] is not None):
            dTotalCount = self._disc_dist_per_group[att_val][self.dTD]

        if(self._disc_dist_per_group[att_val][self.unTD] is not None):
            unTotalCount = self._disc_dist_per_group[att_val][self.unTD]

        if(self._disc_dist_per_group[att_val][self.dGD] is not None):
            dGrantedCount = self._disc_dist_per_group[att_val][self.dGD]

        if(self._disc_dist_per_group[att_val][self.unGD] is not None):
            unGrantedCount = self._disc_dist_per_group[att_val][self.unGD]

        if(dTotalCount != 0):
            dRate = dGrantedCount / dTotalCount

        if(unTotalCount != 0):
            unRate = unGrantedCount / unTotalCount

        val_samples = unTotalCount + dTotalCount

        #return Discrimination Value
        return (abs(dRate - unRate), val_samples)
    



    def update(self, att_val, target_val, w, sen_att_val):
        if att_val is None:
            self._missing_weight_observed += w
        else:
            self._att_values.add(att_val)

            try:
                self._att_dist_per_class[target_val][att_val] += w
            except KeyError:
                self._att_dist_per_class[target_val][att_val] = w

            self.add_disc_dist(att_val, sen_att_val, float(target_val), w)

        self._total_weight_observed += w


    def cond_proba(self, att_val, target_val):
        class_dist = self._att_dist_per_class[target_val]

        if att_val not in class_dist:
            return 0.0

        value = class_dist[att_val]
        try:
            return value / sum(class_dist.values())
        except ZeroDivisionError:
            return 0.0
        
    def calc_disc_per_att(self, weigh=False, normalize=False):
        discrimination = 0.0
        per_att_val_disc = 0.0
        num_att = 0
        num_samples = 0 #use to potentially normalize

        for att_val in self._att_values:
            per_att_val_disc, att_val_num = self.calc_disc_per_val(att_val)
            discrimination += per_att_val_disc
            if weigh:
                discrimination = discrimination*att_val_num #weigh by samples for att
            num_samples += att_val_num #number of total samples
            num_att += 1
        if weigh:
            discrimination = float(discrimination/num_samples) #Normalizing by samples for weighing
        elif normalize:
            discrimination = float(discrimination/len(self._att_values)) #Normalize by number of attribute values

        return discrimination

    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only, node, sens_att_name, trade_off=1):
        best_suggestion = BranchFactory()

        post_disc_merit = self.calc_disc_per_att(weigh=criterion.weigh_discrimination, normalize=criterion.normalize_discrimination)

        if not binary_only:
            post_split_dist = self._class_dist_from_multiway_split()
            merit = criterion.merit_of_split(pre_split_dist, post_split_dist, att_idx, post_disc_merit, node, sens_att_name, trade_off)

            best_suggestion = BranchFactory(
                merit,
                att_idx,
                sorted(self._att_values),
                post_split_dist,
                numerical_feature=False,
                multiway_split=True,
            )

        for att_val in self._att_values:
            post_split_dist = self._class_dist_from_binary_split(att_val)
            merit = criterion.merit_of_split(pre_split_dist, post_split_dist,  att_idx, post_disc_merit, node, sens_att_name, trade_off)
            if best_suggestion is None or merit > best_suggestion.merit:
                best_suggestion = BranchFactory(
                    merit,
                    att_idx,
                    att_val,
                    post_split_dist,
                    numerical_feature=False,
                    multiway_split=False,
                )

        return best_suggestion

    def _class_dist_from_multiway_split(self):
        resulting_dist = {}
        for class_val, att_dist in self._att_dist_per_class.items():
            for att_val, weight in att_dist.items():
                if att_val not in resulting_dist:
                    resulting_dist[att_val] = {}
                if class_val not in resulting_dist[att_val]:
                    resulting_dist[att_val][class_val] = 0.0
                resulting_dist[att_val][class_val] += weight

        sorted_keys = sorted(resulting_dist.keys())
        distributions = [dict(sorted(resulting_dist[k].items())) for k in sorted_keys]
        return distributions

    def _class_dist_from_binary_split(self, val_idx):
        equal_dist = {}
        not_equal_dist = {}
        for class_val, att_dist in self._att_dist_per_class.items():
            for att_val, weight in att_dist.items():
                if att_val == val_idx:
                    if class_val not in equal_dist:
                        equal_dist[class_val] = 0.0
                    equal_dist[class_val] += weight
                else:
                    if class_val not in not_equal_dist:
                        not_equal_dist[class_val] = 0.0
                    not_equal_dist[class_val] += weight
        return [equal_dist, not_equal_dist]