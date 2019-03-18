#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from numpy.testing import assert_array_equal, assert_array_almost_equal
import unittest
import numpy as np

##### Test Classes #####

class TestEventNIScorePredictor(unittest.TestCase):
    def load_data(self):
        """
        load Movielens mini data and generate dummy target
        """
        from pyrecsys.datasets import load_movielens_mini
        data = load_movielens_mini()

        trg = np.empty(data.n_events, dtype=np.int)
        for i in xrange(data.n_events):
            trg[i] = 1 if data.event[i, 1] > 5 else 0

        return data, trg

    def runTest(self):
        import numpy as np
        from inrs.latent_factor import EventNIScorePredictor

        np.random.seed(1234)
        data, trg = self.load_data()

        recommender = EventNIScorePredictor(C=0.01, k=2, eta=100.0)
        recommender.fit(data, trg, disp=False, gtol=1e-03, maxiter=1000)

        self.assertAlmostEqual(recommender.i_loss_, 7.9742820723600563)
        self.assertAlmostEqual(recommender.f_loss_, 0.42314722582423864)

        self.assertAlmostEqual(recommender.predict((1, 7), 0), 4.30083457221)
        self.assertAlmostEqual(recommender.predict((1, 7), 1), 4.00046462398)
        self.assertAlmostEqual(recommender.predict((1, 9), 0), 4.30083464471)
        self.assertAlmostEqual(recommender.predict((1, 9), 1), 4.9984294822)
        self.assertAlmostEqual(recommender.predict((1, 11), 0), 4.30083460989)
        self.assertAlmostEqual(recommender.predict((1, 11), 1), 3.62069734751)
        self.assertAlmostEqual(recommender.predict((3, 7), 0), 3.82129161046)
        self.assertAlmostEqual(recommender.predict((3, 7), 1), 3.72068525262)
        self.assertAlmostEqual(recommender.predict((3, 9), 0), 3.82129165945)
        self.assertAlmostEqual(recommender.predict((3, 9), 1), 4.8901767378)
        self.assertAlmostEqual(recommender.predict((3, 11), 0), 3.82129165943)
        self.assertAlmostEqual(recommender.predict((3, 11), 1), 3.70074539036)
        self.assertAlmostEqual(recommender.predict((5, 7), 0), 2.3010975391)
        self.assertAlmostEqual(recommender.predict((5, 7), 1), 3.72068539498)
        self.assertAlmostEqual(recommender.predict((5, 9), 0), 2.30109759409)
        self.assertAlmostEqual(recommender.predict((5, 9), 1), 4.89017621692)
        self.assertAlmostEqual(recommender.predict((5, 11), 0), 2.3010976029)
        self.assertAlmostEqual(recommender.predict((5, 11), 1), 3.70074526213)

        x = np.array([[1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        assert_array_almost_equal(
            recommender.predict(x, np.zeros(9, dtype=np.int)),
            [4.30083457, 4.30083464, 4.30083461, 3.82129161, 3.82129166,
             3.82129166, 2.30109754, 2.30109759, 2.3010976])
        assert_array_almost_equal(
            recommender.predict(x, np.ones(9, dtype=np.int)),
            [4.00046462, 4.99842948, 3.62069735, 3.72068525, 4.89017674,
             3.70074539, 3.72068539, 4.89017622, 3.70074526])

##### Main routine #####
if __name__ == '__main__':
    unittest.main()
