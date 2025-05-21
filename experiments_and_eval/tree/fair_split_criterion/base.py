from __future__ import annotations

import abc


class SplitCriterion(abc.ABC):
    """SplitCriterion

    Abstract class for computing splitting criteria with respect to distributions of class values.
    The split criterion is used as a parameter on decision trees and decision stumps.

    This class should not me instantiated, as none of its methods are implemented.

    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def merit_of_split(self, pre_split_dist, post_split_dist, attr_name, post_disc_merit, node, sens_att_name):
        """Compute the merit of splitting for a given distribution before the split and after it.

        Parameters
        ----------
        pre_split_dist
            The target statistics before the split.
        post_split_dist
            the target statistics after the split.
        attr_name
            name of the attribute that is splitted
        post_disc_merit
            discrimination after the split
        node
            the node that the split occurs on
        sens_att_name
            the name of the sensitive attribute

        Returns
        -------
        Value of the merit of splitting
        """

    @staticmethod
    @abc.abstractmethod
    def range_of_merit(pre_split_dist):
        """Compute the range of splitting merit.

        Parameters
        ----------
        pre_split_dist
            The target statistics before the split.

        Returns
        -------
        Value of the range of splitting merit
        """