#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate sample data sets
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

__all__ = ['ml100k_year', 'ml100k_gender']

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

def ml100k_year(data, year=1990):
    """
    Generate target variable for the Movielens 100k/1m data set.

    Target variable is 1 if the release year is newer than specified year.

    Parameters
    ----------
    data : pyrecsys.data.EventWithScoreData
        movielens data by pyrecsys.datasets.load_movielens100k or
        load_movielens1m
    year : optional, int
        threshold year, default=1990

    Returns
    -------
    trg : array_like, dtype=int, shape=(n_events,)
        generated target variable
    """
    item_years = data.feature[1][data.event[:, 1]]['year']
    return (np.where(item_years > year, 1, 0))

def ml100k_gender(data):
    """
    Generate target variable for the Movilens 100k/1m data set.

    Target variable is 1 if user is female

    Parameters
    ----------
    data : pyrecsys.data.EventWithScoreData
        movielens data by pyrecsys.datasets.load_movielens100k or
        load_movielens1m

    Returns
    -------
    trg : array_like, dtype=int, shape=(n_events,)
        generated target variable
    """
    return data.feature[0][data.event[:, 0]]['gender']

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
