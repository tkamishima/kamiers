#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recommenders: abstract classes
"""

#==============================================================================
# Module metadata variables
#==============================================================================

#==============================================================================
# Imports
#==============================================================================

import logging
from abc import ABCMeta, abstractmethod
import numpy as np

from pyrecsys.recommenders import BaseEventItemFinder, BaseEventScorePredictor

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['BaseEventNIItemFinder', 'BaseEventNIScorePredictor']

#==============================================================================
# Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

class BaseEventNIItemFinder(BaseEventItemFinder):
    """
    Recommenders to find good items from event data
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseEventNIItemFinder, self).__init__()


class BaseEventNIScorePredictor(BaseEventScorePredictor):
    """
    Recommenders to predict preference scores from event data
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseEventNIScorePredictor, self).__init__()

    @abstractmethod
    def raw_predict(self, ev, **kwargs):
        """
        abstract method: predict score of given one event represented by
        internal ids

        Parameters
        ----------
        eev : array_like, shape=(s_event,) or (n_events, s_event)
            events represented by internal id

        Returns
        -------
        sc : float or array_like, shape=(n_events,), dtype=float
            predicted scores for given inputs
        """
    def predict(self, eev, trg):
        """
        predict score of given event represented by external ids

        Parameters
        ----------
        eev : array_like, shape=(s_event,) or (n_events, s_event)
            events represented by external id
        trg : int or array_like, dtype=int
            target values to enhance information neutrality

        Returns
        -------
        sc : float or array_like, shape=(n_events,), dtype=float
            predicted scores for given inputs
        """

        return self.raw_predict(self.to_iid_event(np.asarray(eev)), trg)


#==============================================================================
# Functions 
#==============================================================================

#==============================================================================
# Module initialization 
#==============================================================================

# init logging system ---------------------------------------------------------

logger = logging.getLogger('pyrecsys')
if not logger.handlers:
    logger.addHandler(logging.NullHandler)

#==============================================================================
# Test routine
#==============================================================================

def _test():
    """ test function for this module
    """

    # perform doctest
    import sys
    import doctest

    doctest.testmod()

    sys.exit(0)

# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
