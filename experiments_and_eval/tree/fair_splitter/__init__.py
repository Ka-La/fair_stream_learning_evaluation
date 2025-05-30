"""
This module implements the Attribute Observers (AO) (or tree splitters) that are used by the
Hoeffding Trees (HT). It also implements the feature quantizers (FQ) used by Stochastic Gradient
Trees (SGT). AOs are a core aspect of the HTs construction, and might represent one of the major
bottlenecks when building the trees. The same holds for SGTs and FQs. The correct choice and setup
of a splitter might result in significant differences in the running time and memory usage of the
incremental decision trees.

AOs for classification and regression trees can be differentiated by using the property
`is_target_class` (`True` for splitters designed to classification tasks). An error will be raised
if one tries to use a classification splitter in a regression tree and vice-versa.
Lastly, AOs cannot be used in SGT and FQs cannot be used in Hoeffding Trees. So, care must be taken
when choosing the correct feature splitter.

"""
from __future__ import annotations

from .base import Splitter
from .gaussian_splitter import GaussianSplitter
from .nominal_splitter_classif import NominalSplitterClassif

__all__ = [
    "GaussianSplitter",
    "NominalSplitterClassif",
    "Splitter"
]