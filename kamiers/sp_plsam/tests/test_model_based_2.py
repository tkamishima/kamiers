#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

from numpy.testing import (
    TestCase,
    run_module_suite,
    assert_,
    assert_allclose,
    assert_array_almost_equal_nulp,
    assert_array_max_ulp,
    assert_array_equal,
    assert_array_less,
    assert_equal,
    assert_raises,
    assert_raises_regex,
    assert_warns,
    assert_string_equal)

import os

import numpy as np

from sklearn.utils import check_random_state

from kamiers.sp_plsam import BaseIndependentMultinomialPLSA
from kamiers.sp_plsam.model_based_2 import IndependentScorePredictor

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestIndependentScorePredictor(TestCase):

    @staticmethod
    def load_data():
        from kamrecsys.datasets import load_event_with_score
        from kamiers.datasets import event_dtype_sensitive_and_timestamp

        infile = os.path.join(os.path.dirname(__file__), 'mlmini_t.event')
        data = load_event_with_score(
            infile, score_domain=(1, 5, 1),
            event_dtype=event_dtype_sensitive_and_timestamp)
        sen = data.event_feature['sensitive']
        return data, sen

    def setUp(self):
        from kamiers.sp_plsam.model_based_2 import (
            IndependentScorePredictor)

        data, sen = self.load_data()
        rec = IndependentScorePredictor(
            k=3, alpha=1.0, random_state=1234, tol=1e-06)
        BaseIndependentMultinomialPLSA.fit(rec, data, sen, event_index=(0, 1))
        rec.score_levels = rec.get_score_levels()

        # setup input data
        sev, ssc, n_events = rec.get_sensitive_divided_data()
        ssc = rec.digitize_score(data, ssc)

        # calc constant parameter
        rec.pS_ = n_events + rec.alpha
        rec.pS_ /= rec.pS_.sum()

        rec.pR_ = (np.asarray(
            [np.bincount(ssc[si], minlength=rec.n_score_levels)
             for si in xrange(rec.n_sensitive_values)]).sum(axis=0) +
                   rec.alpha)
        rec.pR_ /= rec.pR_.sum()

        # random state
        rec._rng = check_random_state(rec.random_state)

        self.rec = rec
        self.params = (sev, ssc, n_events)

    def test___init__(self):
        rec = self.rec

        assert_equal(rec.k, 3)
        assert_allclose(rec.eta, 1.0)
        assert_allclose(rec.alpha, 1.0)
        assert_allclose(rec.optimizer_kwargs['tol'], 1e-06)
        assert_equal(rec.optimizer_kwargs['maxiter'], 100)
        assert_(rec.use_expectation)

        assert_equal(rec.n_objects[0], 8)
        assert_equal(rec.n_objects[1], 10)
        assert_equal(rec.n_events, 30)
        assert_equal(rec.n_sensitive_values, 2)
        assert_equal(rec.n_sensitives, 1)
        assert_equal(rec.n_score_levels, 5)
        assert_allclose(
            rec.score_levels,
            [1.0, 2.0, 3.0, 4.0, 5.0],
            rtol=1e-05)

        assert_equal(rec.method_name, 'plsam_model_based_2')

    def test_loss(self):
        rec = self.rec
        ev, sc, n_events = self.params

        # generate dummy rec._q
        rec._q = np.empty(rec.n_sensitive_values, dtype=object)
        for si in xrange(rec.n_sensitive_values):
            rec._q[si] = np.empty((n_events[si], rec.k), dtype=float)
        rec._q[0][:, 0] = 0.1
        rec._q[0][:, 1] = np.linspace(0.1, 0.8, n_events[0])
        rec._q[0][:, 2] = np.linspace(0.8, 0.1, n_events[0])
        rec._q[1][:, 0] = 0.0
        rec._q[1][:, 1] = 0.5
        rec._q[1][:, 2] = 0.5

        # maximization step
        rec.maximization_step(ev, sc)

        # negative log-likelihood
        assert_allclose(rec.loss(ev, sc), 5.79253960885, rtol=1e-05)

    def test__init_params(self):
        rec = self.rec
        sev, ssc, n_events = self.params
        rec.k = 5
        rec._init_params(sev, ssc)

        assert_allclose(
            rec._q[0][0, :],
            [2.02961910e-04, 9.29037642e-04, 9.98255222e-01, 3.03841990e-04,
             3.08936832e-04],
            rtol=1e-05)
        assert_allclose(
            rec._q[0][4, :],
            [2.63404153e-03, 1.02606552e-03, 4.92867550e-04, 1.51375048e-03,
             9.94333275e-01],
            rtol=1e-05)

        assert_allclose(
            rec._q[1][0, :],
            [1.20441420e-03, 1.59704089e-04, 1.34855369e-03, 9.97124449e-01,
             1.62879011e-04],
            rtol=1e-05)
        assert_allclose(
            rec._q[1][4, :],
            [1.24180236e-04, 3.08835675e-03, 9.92026539e-01, 2.59487742e-03,
             2.16604657e-03],
            rtol=1e-05)

    def test_maximization_step(self):
        rec = self.rec
        sev, ssc, n_events = self.params

        # generate dummy rec._q
        rec._q = np.empty(rec.n_sensitive_values, dtype=object)
        for s in xrange(rec.n_sensitive_values):
            rec._q[s] = np.empty((n_events[s], rec.k), dtype=float)
        rec._q[0][:, 0] = 0.1
        rec._q[0][:, 1] = np.linspace(0.1, 0.8, n_events[0])
        rec._q[0][:, 2] = np.linspace(0.8, 0.1, n_events[0])
        rec._q[1][:, 0] = 0.0
        rec._q[1][:, 1] = 0.5
        rec._q[1][:, 2] = 0.5

        # maximization step
        rec.maximization_step(sev, ssc)

        # Pr[X | S, Z]
        assert_allclose(
            rec.pXgSZ_[0, :, :],
            [[0.17021277, 0.30392684, 0.21355568],
             [0.125, 0.1875, 0.1875]],
            rtol=1e-05)
        assert_allclose(
            rec.pXgSZ_[7, :, :],
            [[0.12765957, 0.11027434, 0.15545992],
             [0.125, 0.125, 0.125]],
            rtol=1e-05)

        # Pr[Y | S, Z]
        assert_allclose(
            rec.pYgSZ_[0, :, :],
            [[0.13157895, 0.19773478, 0.20103823],
             [0.1, 0.05555556, 0.05555556]],
            rtol=1e-05)
        assert_allclose(
            rec.pYgSZ_[9, :, :],
            [[0.0877193, 0.06134969, 0.06134969],
             [0.1, 0.13888889, 0.13888889]],
            rtol=1e-05)

        # Pr[Z | R, S]
        assert_allclose(
            rec.pZgRS_[:, 0, :],
            [[0.33333333, 0.33333333, 0.33333333],
             [0.33333333, 0.33333333, 0.33333333],
             [0.2, 0.40769231, 0.39230769],
             [0.17777778, 0.41111111, 0.41111111],
             [0.2, 0.39230769, 0.40769231]],
            rtol=1e-05)
        assert_allclose(
            rec.pZgRS_[:, 1, :],
            [[0.25, 0.375, 0.375],
             [0.2, 0.4, 0.4],
             [0.2, 0.4, 0.4],
             [0.1, 0.45, 0.45],
             [0.14285714, 0.42857143, 0.42857143]],
            rtol=1e-05)

        # Pr[S]
        assert_allclose(
            rec.pS_,
            [0.46875, 0.53125],
            rtol=1e-05)

        # Pr[R]
        assert_allclose(
            rec.pR_,
            [0.05714286, 0.08571429, 0.2, 0.4, 0.25714286],
            rtol=1e-05)

    def test_fit(self):

        data, sen = self.load_data()
        rec = IndependentScorePredictor(
            k=3, alpha=1.0, random_state=1234, tol=1e-06)
        rec.fit(data, sen)

        assert_equal(rec.fit_results_['n_users'], 8)
        assert_equal(rec.fit_results_['n_items'], 10)
        assert_equal(rec.fit_results_['n_events'], 30)
        assert_equal(rec.fit_results_['n_sensitives'], 1)
        assert_equal(rec.fit_results_['n_sensitive_values'], 2)
        assert_(not rec.fit_results_['success'])
        assert_equal(rec.fit_results_['status'], 2)
        assert_allclose(
            rec.fit_results_['initial_loss'], 5.75311518569751, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['final_loss'], 5.67233229569126, rtol=1e-5)
        assert_allclose(rec.fit_results_['n_iterations'], 100)

        # raw_predict
        assert_allclose(
            np.squeeze(rec.raw_predict(data.event, sen))[:5],
            [3.62537915, 3.70291894, 3.73656522, 3.73576468, 3.71618554],
            rtol=1e-5)

        # predict: corresponding iid = [2, 1]
        assert_allclose(
            rec.predict((5, 2), 0), 3.62537915, rtol=1e-5)

        # predict: corresponding iid = [[7, 6], [2, 0], [7,3]]
        assert_allclose(
            rec.predict([[10, 7], [5, 1], [10, 4]], [1, 0, 0]),
            [3.70291894, 3.73656522, 3.73576468],
            rtol=1e-5)

        # predict with missing values: iid = [[8, 6], [2, 10], [8, 10]]
        assert_allclose(
            rec.predict([[3, 7], [5, 12], [3, 12]], [0, 1, 0]),
            [3.71638698, 3.70243371, 3.71428571],
            rtol=1e-5)

        # use_expectation==False
        rec.use_expectation = False
        assert_array_equal(
            rec.raw_predict(data.event, sen)[:5],
            [4, 4, 4, 4, 5])


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    unittest.main()
