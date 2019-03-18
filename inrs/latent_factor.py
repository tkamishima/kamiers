#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Matrix Decomposition: latent factor model
"""

#==============================================================================
# Module metadata variables
#==============================================================================

#==============================================================================
# Imports
#==============================================================================

import logging
import sys
import copy
import numpy as np
from scipy.optimize import fmin, fmin_powell

from .recommenders import BaseEventNIScorePredictor
from pyrecsys.md import latent_factor

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['EventNIScorePredictor']

#==============================================================================
# Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

class EventNIScorePredictor(BaseEventNIScorePredictor):
    """
    Information neutralized pyrecsys.md.latent_factor.EventScorePredictor

    Parameters
    ----------
    C : float, optional
        regularization parameter (= :math:`\lambda`), default=1.0
    eta : float, optional
        parameter of information neutrality term (= :math:`\eta`),
        default=1.0
    k : int, optional
        the number of latent factors (= sizes of :math:`\mathbf{p}_u` or
        :math:`\mathbf{q}_i`), default=1

    Attributes
    ----------
    `mu_` : array_like
        grobal biases
    `bu_` : array_like
        users' biases
    `bi_` : array_like
        items' biases
    `p_` : array_like
        latent factors of users
    `q_` : array_like
        latent factors of items
    `n_trgvars_` : int
        the numbers of possible target variables
    `i_loss_` : float
        the loss value after initialization
    `f_loss_` : float
        the loss value after fitting
    """

    def __init__(self, C=1.0, eta=1.0, k=1):
        super(EventNIScorePredictor, self).__init__()

        self.C = np.float(C)
        self.eta = np.float(eta)
        self.k = np.int(k)
        self.coef_ = None
        self.mu_ = None
        self.bu_ = None
        self.bi_ = None
        self.p_ = None
        self.q_ = None
        self.n_trgvars_ = 0
        self.i_loss_ = np.inf
        self.f_loss_ = np.inf

    def _init_coef(self, orig_data, tev, tsc, n_objects, **kwargs):
        """
        Initialize model parameters

        Parameters
        ----------
        orig_data : pyrecsys.data.EventWithScoreData
            data before separated depending on vals of target vars
        tev : array, shape(n_events, 2)
            separated event data
        tsc : array, shape(n_events,)
            scores attached to separated events
        n_objects : array, shape(2,)
            vector of numbers of objects
        kwargs : keyword arguments
            parameters for optmizers
        """

        # constants
        n_users = n_objects[0]
        n_items = n_objects[1]
        k = self.k
        n_trgvars = self.n_trgvars_

        # define dtype for parameters
        self._dt = np.dtype([
            ('mu', np.float, (1,)),
            ('bu', np.float, (n_users,)),
            ('bi', np.float, (n_items,)),
            ('p', np.float, (n_users, k)),
            ('q', np.float, (n_items, k))
        ])

        # memory allocation
        dt_itemsize = 1 + n_users + n_items +\
                      n_users * k + n_items * k
        self.coef_ = np.zeros(dt_itemsize * n_trgvars, dtype=np.float)

        # set array's view
        self.mu_ = self.coef_.view(self._dt)['mu']
        self.bu_ = self.coef_.view(self._dt)['bu']
        self.bi_ = self.coef_.view(self._dt)['bi']
        self.p_ = self.coef_.view(self._dt)['p']
        self.q_ = self.coef_.view(self._dt)['q']

        # init model by normal recommenders
        data = copy.copy(orig_data)
        recommender = latent_factor.EventScorePredictor(C=self.C, k=k)
        for t in xrange(n_trgvars):
            data.event = tev[t]
            data.n_events = tev[t].shape[0]
            data.score = tsc[t]
            recommender.fit(data, **kwargs)

            self.mu_[t][:] = recommender.mu_
            self.bu_[t][:] = recommender.bu_[:n_users]
            self.bi_[t][:] = recommender.bi_[:n_items]
            self.p_[t][:, :] = recommender.p_[:n_users, :]
            self.q_[t][:, :] = recommender.q_[:n_items, :]

        # scale a regularization term by the number of parameters
        self._reg = self.C / np.float((dt_itemsize - 2 - 2 * k) * n_trgvars)

    def loss(self, coef, tev, tsc, n_objects, score_bins):
        """
        loss function to optimize

        main loss function: same as the pyrecsys.md.latent_factor.

        information neutrality term:

        To estimate distribution of estimated scores, we adopt a histogram
        model. Estimated scores are first discretized into bins according to
        `score_bins`. Distributions are derived from the frequencies of
        events in these bins.

        \sum_{u, i, t} \sum_{s \in bins} \
            I(bin(\hat{s}(u, i, t)) == s) log p[s | t] - log p[s]

        u, i, t are user_index, item_index, information neutral target var,
        respectively. \hat(s)(u, i, t) is estimated score, bin(s) is a
        function that returns index of the bin to which s belongs.
        Distribution p[y | t] is derived by generating histogram from a data
        set that is consists of events whose target values equal to t.
        p[s] = \sum_t \p[t] p[s | t].

        Parameters
        ----------
        coef : array_like, shape=(variable,)
            coefficients of this model
        tev : array_like, shape(n_events, 2), dtype=int
            user and item indexes
        tsc : array_like, shape(n_events,), dtype=float
            target scores
        n_objects : array_like, shape(2,), dtype=int
            numbers of users and items
        score_bins : array_like, shape=(variable,) dtype=float, optional
            thresholds to discretize scores.

        Returns
        -------
        loss : float
            value of loss function
        """
        # constants
        n_events = np.array([ev.shape[0] for ev in tev])
        n_users = n_objects[0]
        n_items = n_objects[1]
        k = self.k

        # set array's view
        mu = coef.view(self._dt)['mu']
        bu = coef.view(self._dt)['bu']
        bi = coef.view(self._dt)['bi']
        p = coef.view(self._dt)['p']
        q = coef.view(self._dt)['q']

        # basic stats
        esc = np.empty(self.n_trgvars_, dtype=object)
        pesct = np.empty((self.n_trgvars_, len(score_bins) - 1),
                         dtype=np.float)
        for t in xrange(self.n_trgvars_):
            ev = tev[t]
            esc[t] = mu[t][0] + bu[t][ev[:, 0]] + bi[t][ev[:, 1]] +\
                     np.sum(p[t][ev[:, 0], :] * q[t][ev[:, 1], :], axis=1)
            pesct[t, :] = np.histogram(esc[t], score_bins)[0]
        pesc = np.sum(pesct, axis=0) / np.sum(n_events)
        pesct = pesct / n_events[:, np.newaxis]

        # loss term and information neutrality term
        loss = 0.0
        for t in xrange(self.n_trgvars_):
            loss += np.sum((tsc[t] - esc[t]) ** 2)

        # information neutrality term
        in_term = 0.0
        for t in xrange(self.n_trgvars_):
            # NOTE: to avoid 0 in log
            pos = np.nonzero(pesct[t, :] > 0.0)[0]
            in_term += np.dot(pesct[t, pos],
                              np.log(pesct[t, pos]) - np.log(pesc[pos])) *\
                       n_events[t]

        # regularization term
        reg = 0.0
        for t in xrange(self.n_trgvars_):
            reg += np.sum(bu[t] ** 2) + np.sum(bi[t] ** 2) +\
                   np.sum(p[t] ** 2) + np.sum(q[t] ** 2)

        return loss / np.float(np.sum(n_events)) +\
               self.eta * in_term / np.float(np.sum(n_events)) +\
               self._reg * reg

    def fit(self, data, trg, n_trgvars=2,
            user_index=0, item_index=1, score_index=0,
            score_bins=None, **kwargs):
        """
        fitting model

        Parameters
        ----------
        data : :class:`pyrecsys.data.EventWithScoreData`
            data to fit
        trg : array_like, shape=(n_events,)
            a variable to be neutral in recommendation
        n_trgvars : int
            the numbers of possible target variables
        user_index : optional, int
            Index to specify the position of a user in an event vector.
            (default=0)
        item_index : optioanl, int
            Index to specify the position of a item in an event vector.
            (default=1)
        score_index : optional, int
            Ignored if score of data is a single criterion type. In a multi-
            criteria case, specify the position of the target score in a score
            vector. (default=0)
        score_bins : array_like, ndim=1
            a vector of thresholds to separate scores to bins
            default=[-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf],
            for 1 to 5 rating scores
        kwargs : keyowrd arguments
            keyword arguments passed to optimizers
        """

        # set parameters
        self.n_trgvars_ = n_trgvars
        if score_bins is None:
            score_bins = np.array([-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf])

        # get input data
        ev, sc, n_objects =\
        self._get_event_and_score(data,
            (user_index, item_index), score_index)

        # divide events and scores according to the corresponding target
        # variables
        tev = np.empty(self.n_trgvars_, dtype=np.object)
        tsc = np.empty(self.n_trgvars_, dtype=np.object)
        for t in xrange(self.n_trgvars_):
            tev[t] = ev[trg == t, :]
            tsc[t] = sc[trg == t]

        # initialize coefficients
        self._init_coef(data, tev, tsc, n_objects,
                        gtol=kwargs.pop('gtol', 1e-05),
                        maxiter=kwargs.get('maxiter', None),
                        disp=kwargs.get('disp', False))

        # check optimization parameters
        if not 'disp' in kwargs:
            kwargs['disp'] = False
        if 'gtol' in kwargs:
            del kwargs['gtol']

        # get final loss
        self.i_loss_ = self.loss(self.coef_, tev, tsc, n_objects, score_bins)

        # optimize model
        self.coef_[:] = fmin_powell(self.loss,
                                    self.coef_,
                                    args=(tev, tsc, n_objects, score_bins),
                                    **kwargs)

        # add parameters for unknown users and items
        self.bu_ = np.empty(self.n_trgvars_, dtype=object)
        self.bi_ = np.empty(self.n_trgvars_, dtype=object)
        self.p_ = np.empty(self.n_trgvars_, dtype=object)
        self.q_ = np.empty(self.n_trgvars_, dtype=object)
        for t in xrange(self.n_trgvars_):
            self.bu_[t] = np.r_[self.coef_.view(self._dt)['bu'][t], 0.0]
            self.bi_[t] = np.r_[self.coef_.view(self._dt)['bi'][t], 0.0]
            self.p_[t] = np.r_[self.coef_.view(self._dt)['p'][t],
                               np.zeros((1, self.k), dtype=np.float)]
            self.q_[t] = np.r_[self.coef_.view(self._dt)['q'][t],
                               np.zeros((1, self.k), dtype=np.float)]

        # get final loss
        self.f_loss_ = self.loss(self.coef_, tev, tsc, n_objects, score_bins)

        # clean up temporary instance variables
        del self.coef_
        del self._reg
        del self._dt

    def raw_predict(self, ev, trg):
        """
        predict score of given one event represented by internal ids

        Parameters
        ----------
        (user, item) : array_like
            a target user's and item's ids. unknwon objects assumed to be
            represented by n_object[event_otype]

        Returns
        -------
        ev : array_like, shape=(s_events,) or (variable, s_events)
            events for which scores are predicted
        trg : int or array_like, dtype=int
            target values to enhance information neutrality

        Raises
        ------
        TypeError
            shape of an input array is illegal
        """

        if ev.ndim == 1:
            return self.mu_[trg][0] +\
                   self.bu_[trg][ev[0]] + self.bi_[trg][ev[1]] +\
                   np.dot(self.p_[trg][ev[0]], self.q_[trg][ev[1]])
        elif ev.ndim == 2:
            return np.array([self.mu_[t][0] +\
                             self.bu_[t][ev[i, 0]] + self.bi_[t][ev[i, 1]] +\
                             np.dot(self.p_[t][ev[i, 0], :],
                                    self.q_[t][ev[i, 1], :])
                             for i, t in enumerate(trg)])
        else:
            raise TypeError

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
