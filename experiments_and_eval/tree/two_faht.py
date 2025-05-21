from __future__ import annotations

import random

from river import base, drift
from river.tree.nodes.branch import DTBranch
from river.utils.norm import normalize_values_in_dict

from tree.nodes.leaf import HTLeaf

from .faht import HoeffdingTreeClassifier
from .nodes.branch import DTBranch
from .nodes.hatc_nodes import (
    AdaBranchClassifier,
    AdaLeafClassifier,
    AdaNomBinaryBranchClass,
    AdaNomMultiwayBranchClass,
    AdaNumBinaryBranchClass,
    AdaNumMultiwayBranchClass,
)
from .fair_splitter import Splitter
from river.tree.utils import add_dict_values


class HoeffdingAdaptiveTreeClassifier(HoeffdingTreeClassifier):
    """Hoeffding Adaptive Tree classifier.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow until
          the system recursion limit.
    split_criterion
        Split criterion to use.</br>
        - 'fair_info_gain' - Fair Information Gain</br>
        - 'fair_enhancing_info_gain' - Fair Enhancing Information Gain</br>
        - 'flex_info_gain' - Flexible Fair Information Gain</br>
    delta
        Significance level to calculate the Hoeffding bound. The significance level is given by
        `1 - delta`. Values closer to zero imply longer split decision delays.
    tau
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mc' - Majority Class</br>
        - 'nb' - Naive Bayes</br>
        - 'nba' - Naive Bayes Adaptive</br>
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes. If empty, then assume that all numeric attributes should
        be treated as continuous.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.GaussianSplitter` is used if `splitter` is `None`.
    bootstrap_sampling
        If True, perform bootstrap sampling in the leaf nodes.
    drift_window_threshold
        Minimum number of examples an alternate tree must observe before being considered as a
        potential replacement to the current one.
    drift_detector
        The drift detector used to build the tree. If `None` then `drift.DummyDriftDetector` is used.
    drift_t
        The window after which we check for drift if the drift detector is a fixed dummy, should be bigger than drift_window_threshold!
    switch_significance
        The significance level to assess whether alternate subtrees are significantly better
        than their main subtree counterparts.
    binary_split
        If True, only allow binary splits.
    min_branch_fraction
        The minimum percentage of observed data required for branches resulting from split
        candidates. To validate a split candidate, at least two resulting branches must have
        a percentage of samples greater than `min_branch_fraction`. This criterion prevents
        unnecessary splits when the majority of instances are concentrated in a single branch.
    max_share_to_split
        Only perform a split in a leaf if the proportion of elements in the majority class is
        smaller than this parameter value. This parameter avoids performing splits when most
        of the data belongs to a single class.
    max_size
        The max size of the tree, in mebibytes (MiB).
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    merit_preprune
        If True, enable merit-based tree pre-pruning.
    seed
       Random seed for reproducibility.


    Notes
    -----
    The Hoeffding Adaptive Tree [^1] uses a drift detector to monitor performance of branches in
    the tree and to replace them with new branches when their accuracy decreases.

    The bootstrap sampling strategy is an improvement over the original Hoeffding Adaptive Tree
    algorithm. It is enabled by default since, in general, it results in better performance.

    References
    ----------
    [^1]: Bifet, Albert, and Ricard Gavaldà. "Adaptive learning from evolving data streams."
       In International Symposium on Intelligent Data Analysis, pp. 249-260. Springer, Berlin,
       Heidelberg, 2009.

    Examples
    --------
    >>> from river.datasets import synth
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> gen = synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=0),
    ...                                drift_stream=synth.SEA(seed=42, variant=1),
    ...                                seed=1, position=500, width=50)
    >>> # Take 1000 instances from the infinite data generator
    >>> dataset = iter(gen.take(1000))

    >>> model = tree.HoeffdingAdaptiveTreeClassifier(
    ...     grace_period=100,
    ...     delta=1e-5,
    ...     leaf_prediction='nb',
    ...     nb_threshold=10,
    ...     seed=0
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 91.49%

    """

    def __init__(
        self,
        sensitive_attribute: str, #TODO: Consider whether a default - like "sex" - makes sense here
        deprived_idx: str | int, #Idx of deprived group
        fairness_criteria: str = "demographic_parity",
        trade_off: int = 1, #the trade-off between info and fairness gain in flexible mode
        grace_period: int = 200,
        max_depth: int | None = None,
        split_criterion: str = "flex_info_gain",
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        bootstrap_sampling: bool = True,
        drift_window_threshold: int = 300,
        drift_detector: base.DriftDetector | None = None,
        drift_t: int = 330,
        switch_significance: float = 0.05,
        binary_split: bool = False,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        max_size: float = 100.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            sensitive_attribute=sensitive_attribute,
            deprived_idx=deprived_idx,
            trade_off=trade_off,
            grace_period=grace_period,
            max_depth=max_depth,
            split_criterion=split_criterion,
            delta=delta,
            tau=tau,
            leaf_prediction=leaf_prediction,
            nb_threshold=nb_threshold,
            nominal_attributes=nominal_attributes,
            splitter=splitter,
            binary_split=binary_split,
            min_branch_fraction=min_branch_fraction,
            max_share_to_split=max_share_to_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

        self.fairness_criteria = fairness_criteria

        self.bootstrap_sampling = bootstrap_sampling
        self.drift_window_threshold = drift_window_threshold
        self.drift_t = drift_t #This is done because paper describes use of periodic checks instead of use of ADWIN. If ADWIN should be used, pass it on!
        if(not self.drift_t > self.drift_window_threshold):
            print("drift_t is chosen smaller than min. drift window; alt trees will never be used this way!!!")
        self.drift_detector = drift_detector if drift_detector is not None else drift.DummyDriftDetector(trigger_method="fixed", t_0 =self.drift_t)
        self.switch_significance = switch_significance
        self.seed = seed

        self._n_alternate_trees = 0
        self._n_pruned_alternate_trees = 0
        self._n_switch_alternate_trees = 0

        self._rng = random.Random(self.seed)

    @property
    def _mutable_attributes(self):
        return {"grace_period", "delta", "tau", "drift_window_threshold", "switch_significance"}

    @property
    def n_alternate_trees(self):
        return self._n_alternate_trees

    @property
    def n_pruned_alternate_trees(self):
        return self._n_pruned_alternate_trees

    @property
    def n_switch_alternate_trees(self):
        return self._n_switch_alternate_trees

    @property
    def summary(self):
        summ = super().summary
        summ.update(
            {
                "n_alternate_trees": self.n_alternate_trees,
                "n_pruned_alternate_trees": self.n_pruned_alternate_trees,
                "n_switch_alternate_trees": self.n_switch_alternate_trees,
            }
        )
        return summ

    def learn_one(self, x, y, *, w=1.0):
        # Updates the set of observed classes
        self.classes.add(y)

        self._train_weight_seen_by_model += w

        if self._root is None:
            self._root = self._new_leaf()
            self._n_active_leaves = 1

        self._root.learn_one(x, y, w=w, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

    # Override HoeffdingTreeClassifier
    def predict_proba_one(self, x):
        proba = {c: 0.0 for c in self.classes}
        if self._root is not None:
            found_nodes = [self._root]
            if isinstance(self._root, DTBranch):
                found_nodes = self._root.traverse(x, until_leaf=True)
            for leaf in found_nodes:
                dist = leaf.prediction(x, tree=self)
                # Option Tree prediction (of sorts): combine the response of all leaves reached
                # by the instance
                proba = add_dict_values(proba, dist, inplace=True)
            proba = normalize_values_in_dict(proba)

        return proba

    def _new_leaf(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}

        if parent is not None:
            depth = parent.depth + 1
        else:
            depth = 0

        return AdaLeafClassifier(
            sens_att=self.sensitive_attribute,
            sens_idx=self.deprived_idx,
            fairness_criteria="statistical_parity",
            stats=initial_stats,
            depth=depth,
            splitter=self.splitter,
            drift_detector=self.drift_detector.clone(),
            rng=self._rng,
        )

    def _branch_selector(
        self, numerical_feature=True, multiway_split=False
    ) -> type[AdaBranchClassifier]:
        """Create a new split node."""
        if numerical_feature:
            if not multiway_split:
                return AdaNumBinaryBranchClass
            else:
                return AdaNumMultiwayBranchClass
        else:
            if not multiway_split:
                return AdaNomBinaryBranchClass
            else:
                return AdaNomMultiwayBranchClass
            
    def _attempt_to_split(self, leaf: HTLeaf, parent: DTBranch, parent_branch: int, **kwargs):
        """Attempt to split a leaf.

        If the samples seen so far are not from the same class then:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the difference between the top 2 split candidates is larger than the Hoeffding bound:
           3.1 Replace the leaf node by a split node (branch node).
           3.2 Add a new leaf node on each branch of the new split node.
           3.3 Update tree's metrics

        Optional: Disable poor attributes. Depends on the tree's configuration.

        Parameters
        ----------
        leaf
            The leaf to evaluate.
        parent
            The leaf's parent.
        parent_branch
            Parent leaf's branch index.
        kwargs
            Other parameters passed to the new branch.
        """
        if not leaf.observed_class_distribution_is_pure():  # type: ignore
            split_criterion = self._new_split_criterion()

            best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort()
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(leaf.stats),
                    self.delta,
                    leaf.total_weight,
                )
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (
                    best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                    or hoeffding_bound < self.tau
                ):
                    should_split = True
                if self.remove_poor_attrs:
                    poor_atts = set()
                    # Add any poor attribute to set
                    for suggestion in best_split_suggestions:
                        if (
                            suggestion.feature
                            and best_suggestion.merit - suggestion.merit > hoeffding_bound
                        ):
                            poor_atts.add(suggestion.feature)
                    for poor_att in poor_atts:
                        leaf.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
                if split_decision.feature is None:
                    # Pre-pruning - null wins
                    leaf.deactivate()
                    self._n_inactive_leaves += 1
                    self._n_active_leaves -= 1
                else:
                    branch = self._branch_selector(
                        split_decision.numerical_feature, split_decision.multiway_split
                    )
                    leaves = tuple(
                        self._new_leaf(initial_stats, parent=leaf)
                        for initial_stats in split_decision.children_stats  # type: ignore
                    )

                    kwargs['sens_att'] = self.sensitive_attribute
                    kwargs['sens_idx'] = self.deprived_idx
                    kwargs['fairness_criteria'] = self.fairness_criteria

                    new_split = split_decision.assemble(
                        branch, leaf.stats, leaf.depth, *leaves, **kwargs
                    )

                    self._n_active_leaves -= 1
                    self._n_active_leaves += len(leaves)
                    if parent is None:
                        self._root = new_split
                    else:
                        parent.children[parent_branch] = new_split

                # Manage memory
                self._enforce_size_limit()


    @classmethod
    def _unit_test_params(cls):
        yield {"seed": 1}