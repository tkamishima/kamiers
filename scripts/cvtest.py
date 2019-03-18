#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
do cross validation on movielens whose sensitive variable is release year
of movies.

SYNOPSIS::

    SCRIPT [options]

Output
------
following values are output to <stdout>

1. user's external ID
2. item's external ID
3. a value of a sensitive variable
4. a true rating value
5. an expected rating value

Options
-------
-i <INPUT>, --in <INPUT>
    specify <INPUT> file stem
-C <C>, --lambda <C>
    regularization parameter, default=0.01.
-e <ETA>, --eta <ETA>
    weight of neutrality parameter, default=0.01
-k <K>, --dim <K>
    the number of dimensions of latent, default=1.
-t <TOL>, --tol <TOL>
    optimization parameter. the size of norm of gradient. default=1e-03.
-s <SENSITIVE>, --sensitive <SENSITIVE>
    [gender|year] type of sensitive variable, default=gender.
-q, --quiet
    set logging level to ERROR, no messages unless errors
--rseed <RSEED>
    random number seed. if None, use /dev/urandom (default None)
--version
    show version
"""

#==============================================================================
# Imports
#==============================================================================

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import sys
import argparse
import os
import platform
import commands
import logging
import datetime
import numpy as np

# private modules
import site

site.addsitedir('.')

from pyrecsys.datasets import load_movielens100k
from inrs.datasets import ml100k_year, ml100k_gender
from inrs.latent_factor import EventNIScorePredictor

#==============================================================================
# Module metadata variables
#==============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2012/04/17"
__version__ = "1.0.0"
__copyright__ = "Copyright (c) 2012 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"

#==============================================================================
# Public symbols
#==============================================================================

__all__ = []

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

def eval(opt, fold):
    """
    Do the specified fold
    """

    ### pre process
    data = load_movielens100k('%s@%1dl.event' % (opt.stem, fold))
    if opt.sensitive == 'gender':
        trg = ml100k_gender(data)
    elif opt.sensitive == 'year':
        trg = ml100k_year(data)
    else:
        raise ValueError('invalid sensitive variable')

    ### main process

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    logger.info("start time[%02d] = %s" % (fold, start_time.isoformat()))

    # main process
    recommender = EventNIScorePredictor(C=opt.C, k=opt.k, eta=opt.eta)
    recommender.fit(data, trg, gtol=opt.tol, xtol=opt.tol, ftol=opt.tol)

    # set end and elapsed time
    end_time = datetime.datetime.now()
    end_utime = os.times()[0]
    logger.info("end_time[%02d] = %s" % (fold, end_time.isoformat()))
    logger.info("elapsed_time[%02d] = %s" % (fold,
                                             str(end_time - start_time)))
    vars(opt)['elapsed_time[%02d]' % fold] = str((end_time - start_time))
    logger.info("elapsed_utime[%02d] = %s" % (fold,
                                              str(end_utime - start_utime)))
    vars(opt)['elapsed_utime[%02d]' % fold] = str((end_utime - start_utime))
    logger.info("loss: " + str(recommender.i_loss_) +
                " => " + str(recommender.f_loss_))
    vars(opt)['init_loss[%02d]' % fold] = str(recommender.i_loss_)
    vars(opt)['final_loss[%02d]' % fold] = str(recommender.f_loss_)

    # prediction
    data = load_movielens100k('%s@%1dt.event' % (opt.stem, fold))
    ev = data.to_eid_event(data.event)
    if opt.sensitive == 'gender':
        trg = ml100k_gender(data)
    elif opt.sensitive == 'year':
        trg = ml100k_year(data)
    else:
        raise ValueError('invalid sensitive variable')

    ### output

    # output evaluation results
    for i in xrange(data.n_events):
        sys.stdout.write(str(ev[i, 0]) + "\t")
        sys.stdout.write(str(ev[i, 1]) + "\t")
        sys.stdout.write(str(trg[i]) + "\t")
        sys.stdout.write(str(data.score[i]) + "\t")
        esc = recommender.predict(ev[i, :], trg[i])
        sys.stdout.write(str(esc) + "\n")

    # output option information
    for key, key_val in vars(opt).iteritems():
        sys.stdout.write("#%s=%s\n" % (key, str(key_val)))

#==============================================================================
# Main routine
#==============================================================================

if __name__ == '__main__':
    ### set script name
    script_name = sys.argv[0].split('/')[-1]

    ### init logging system
    logger = logging.getLogger(script_name)
    logging.basicConfig(level=logging.INFO,
                        format='[%(name)s: %(levelname)s'
                               ' @ %(asctime)s] %(message)s')

    ### command-line option parsing

    ap = argparse.ArgumentParser(
        description='pydoc is useful for learning the details.')

    # common options
    ap.add_argument('--version', action='version',
                    version='%(prog)s ' + __version__)

    apg = ap.add_mutually_exclusive_group()
    apg.set_defaults(verbose=True)
    apg.add_argument('--verbose', action='store_true')
    apg.add_argument('-q', '--quiet', action='store_false', dest='verbose')

    ap.add_argument("--rseed", type=int, default=None)

    # script specific options
    ap.add_argument('-i', '--in', type=str, default='00DATA/mlsmall',
                    dest='stem')
    ap.add_argument('-C', '--lambda', dest='C', type=float, default=0.01)
    ap.add_argument('-e', '--eta', dest='eta', type=float, default=0.01)
    ap.add_argument('-k', '--dim', dest='k', type=int, default=1)
    ap.add_argument('-t', '--tol', dest='tol', type=float, default=1e-03)
    ap.add_argument('-f', '--fold', dest='fold', type=int, default=5)
    ap.add_argument('-s', '--sensitive', dest='sensitive', type=str,
                    default='gender', choices=['gender', 'year'])

    # parsing
    opt = ap.parse_args()

    # post-processing for command-line options
    # disable logging messages by changing logging level
    if not opt.verbose:
        logger.setLevel(logging.ERROR)

    # set random seed
    np.random.seed(opt.rseed)

    ### set meta-data of script and machine
    opt.script_name = script_name
    opt.script_version = __version__
    opt.python_version = platform.python_version()
    opt.sys_uname = platform.uname()
    if platform.system() == 'Darwin':
        opt.sys_info =\
        commands.getoutput('system_profiler'
                           ' -detailLevel mini SPHardwareDataType')\
        .split('\n')[4:-1]
    elif platform.system() == 'FreeBSD':
        opt.sys_info = commands.getoutput('sysctl hw').split('\n')
    elif platform.system() == 'Linux':
        opt.sys_info = commands.getoutput('cat /proc/cpuinfo').split('\n')

    ### suppress warnings in numerical computation
    np.seterr(all='ignore')

    ### call main routine
    for i in xrange(opt.fold):
        eval(opt, i)










