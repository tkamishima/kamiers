#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summary of __THIS_MODULE__
"""

#==============================================================================
# Module metadata variables
#==============================================================================

#==============================================================================
# Imports
#==============================================================================

import logging
import numpy as np

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['mi_disc']

#==============================================================================
# Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

#==============================================================================
# Functions
#==============================================================================

def mi_disc(sc, trg, score_bins, n_trgvars):
    """
    Mutual Information between discretized scores and target values

    Parameters
    ----------
    sc : array_like, shape=(n_events,), dtype=np.float
        scores
    trg : array_like, shape=(n_events,), dtype=np.float
        target values
    score_bins : array_like, ndim=1
        a vector of thresholds to separate scores to bins
        default=[-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf],
        for 1 to 5 rating scores
    n_trgvars : int
            the numbers of possible target variables

    Returns
    -------
    mi : float
        I(SC; TRG) = H(SC) + H(TRG) - H(SC, TRG). mutual information
    mi_per_sc, mi_per_trg : float
        I(SC; TRG) / H(SC) and I(SC; TRG) / H(TRG)
        normalized by entropies of scores or target values
    a_mean, g_mean : float
        Arithmetic and geometric means of mi_per_sc and mi_per_trg
    """

    # joint entropy of the pmf function n / sum(n)
    en = lambda n: np.sum([0.0 if i == 0.0
                           else (-i / np.float(np.sum(n))) *\
                                np.log2(i / np.float(np.sum(n)))
                           for i in np.ravel(n)])

    n_bins = len(score_bins) - 1
    hist = np.empty((n_bins, n_trgvars), dtype=np.int)
    for t in xrange(n_trgvars):
        hist[:, t] = np.histogram(sc[trg == t], score_bins)[0]

    en_sc = en(np.sum(hist, axis=1))
    en_trg = en(np.sum(hist, axis=0))
    en_jnt = en(hist)

    mi = np.max((0.0, en_sc + en_trg - en_jnt))
    mi_per_sc = 1.0 if en_sc <= 0.0 else mi / en_sc
    mi_per_trg = 1.0 if en_trg <= 0.0 else mi / en_trg

    return mi, mi_per_sc, mi_per_trg,\
           (mi_per_sc + mi_per_trg) / 2.0,\
           np.sqrt(mi_per_sc * mi_per_trg)

#==============================================================================
# Module initialization
#==============================================================================

# init logging system

logger = logging.getLogger('inrs')
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

# Check if this is call as command script

if __name__ == '__main__':
    _test()
