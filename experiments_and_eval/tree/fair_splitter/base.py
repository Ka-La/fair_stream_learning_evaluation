from __future__ import annotations

import abc
import typing
from ..nodes.leaf import HTLeaf

from river import base

from ..fair_split_criterion.base import SplitCriterion
from river.tree.utils import BranchFactory, GradHess, GradHessStats


class Splitter(base.Estimator, abc.ABC):
    """Base class for the tree splitters.

    Each Attribute Observer (AO) or Splitter monitors one input feature and finds the best
    split point for this attribute. AOs can also perform other tasks related to the monitored
    feature, such as estimating its probability density function (classification case).

    This class should not be instantiated, as none of its methods are implemented.

    """

    @abc.abstractmethod
    def update(self, att_val, target_val: base.typing.Target, w: float, sen_att_val):
        """Update statistics of this observer given an attribute value, its target value
        and the weight of the instance observed.

        Parameters
        ----------
        att_val
            The value of the monitored attribute.
        target_val
            The target value.
        w
            The weight of the instance.
        sen_att_val 
            The value of the sensitive/protected attribute

        """

    @abc.abstractmethod
    def cond_proba(self, att_val, target_val: base.typing.ClfTarget) -> float:
        """Get the probability for an attribute value given a class.

        Parameters
        ----------
        att_val
            The value of the attribute.
        target_val
            The target (class label) value.

        Returns
        -------
            Probability for an attribute value given a class.

        """

    @abc.abstractmethod
    def best_evaluated_split_suggestion(
        self,
        criterion: SplitCriterion,
        pre_split_dist: list | dict,
        att_idx: base.typing.FeatureName,
        binary_only: bool,
        node: HTLeaf,
        sens_att_name: str,
        trade_off: int, 
    ) -> BranchFactory:
        """Get the best split suggestion given a criterion and the target's statistics.

        Parameters
        ----------
        criterion
            The split criterion to use.
        pre_split_dist
            The target statistics before the split.
        att_idx
            The attribute index.
        binary_only
            True if only binary splits are allowed.
        node
            The node on which the split is performed.
        sens_att_name
            The name of the sensitive attribute.
        trade_off
            If Fairness Gain is computed flexibly, this is the trade-off hyperparameter

        Returns
        -------
            Suggestion of the best attribute split.
        """

    @property
    def is_numeric(self) -> bool:
        """Determine whether or not the splitter works with numerical features."""
        return True

    @property
    def is_target_class(self) -> bool:
        """Check on which kind of learning task the splitter is designed to work.

        If `True`, the splitter works with classification trees, otherwise it is designed for
        regression trees.
        """
        return True

