#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)
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
import numpy as np

from kamrecsys.datasets import load_movielens_mini

from kamiers.base import (
    BaseIndependentScorePredictorFromSingleBinarySensitive)

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class IndependentScorePredictorFromSingleBinarySensitive(
        BaseIndependentScorePredictorFromSingleBinarySensitive):

    def __init__(self):
        super(IndependentScorePredictorFromSingleBinarySensitive,
              self).__init__()

    def raw_predict(self, ev, sen):

        return np.where(sen == 0, 1, 0)


class TestBaseIndependentScorePredictorFromSingleBinarySensitive(TestCase):

    def test_predict(self):

        data = load_movielens_mini()
        rec = IndependentScorePredictorFromSingleBinarySensitive()
        s_orig = np.r_[np.zeros(15), np.ones(15)]

        # fit()
        rec.fit(data, s_orig, event_index=(0, 1))

        assert_array_equal(rec.sensitive, s_orig)
        assert_equal(rec.n_sensitives, 1)
        assert_equal(rec.n_sensitive_values, 2)

        # get_sensitive()
        s, s_s, n_s_values = rec.get_sensitive()
        assert_equal(s.ndim, 1)
        assert_equal(s.shape, (30,))
        assert_array_equal(s, s_orig)
        assert_equal(s_s, 1)
        assert_equal(n_s_values, 2)

        # get_sensitive_divided_data
        sev, ssc, n_events = rec.get_sensitive_divided_data()
        assert_array_equal(data.event[:15, :], sev[0])
        assert_array_equal(data.event[15:, :], sev[1])
        assert_array_equal(n_events, [15, 15])
        assert_allclose(data.score[:15], ssc[0])
        assert_allclose(data.score[15:], ssc[1])
        # check original data event is not destroyed
        assert_array_equal(
            rec.event[:3, :],
            rec.to_iid_event(np.array([[5, 2], [10, 7], [5, 1]])))
        assert_array_equal(
            rec.event[15:18, :],
            rec.to_iid_event(np.array([[1, 8], [1, 1], [2, 10]])))
        assert_allclose(rec.score[-3:], [4, 4, 4])

        # predict()
        assert_equal(rec.predict([0, 0], 0), 1)
        assert_equal(rec.predict([0, 0], 1), 0)
        assert_array_equal(
            rec.predict([[0, 1],  [1, 0]], [0, 1]), [1, 0])

        # remove_data
        rec.remove_data()
        assert_(rec.sensitive is None)


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
