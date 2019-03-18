#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from numpy.testing import assert_array_equal, assert_array_almost_equal
import unittest

##### Test Classes #####

class TestDataSets(unittest.TestCase):

    def test_ml100k_year(self):
        import numpy as np
        from pyrecsys.datasets import load_movielens_mini
        from inrs.datasets import ml100k_year

        data = load_movielens_mini()
        assert_array_equal(
            ml100k_year(data, 1995),
            np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        assert_array_equal(ml100k_year(data), np.ones(30))
        assert_array_equal(ml100k_year(data, year=2000), np.zeros(30))

    def test_ml100k_gender(self):
        import numpy as np
        from pyrecsys.datasets import load_movielens_mini
        from inrs.datasets import ml100k_gender

        data = load_movielens_mini()
        assert_array_equal(
            ml100k_gender(data),
            np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

##### Main routine #####
if __name__ == '__main__':
    unittest.main()
