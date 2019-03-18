#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calc statistics

SYNOPSIS::

    SCRIPT <input_file>

Input Format
------------

tab separated. each column contains

1. external user ID
2. external item ID
3. target value
4. true score
5. estimated score

Output Format
-------------

tab separated. each column contains

1. file name
2. MAE of scores
3. RMSE of scores
4. I(SC; TRG)
5. I(SC; TRG) / H(SC)
6. I(SC; TRG) / H(TRG)
7. An arithmetic mean of 5 and 6
8. An geometric mean of 5 and 6
"""

import sys
import numpy as np
from scipy.spatial.distance import minkowski, sqeuclidean
from inrs.eval import mi_disc

infile = sys.argv[1]
dt = np.dtype([
    ('event', np.int, 2),
    ('view', np.int, 1),
    ('tsc', np.float, 1),
    ('esc', np.float, 1)
])
data = np.genfromtxt(infile, delimiter='\t', dtype=dt)

n_samples = len(data['tsc'])
mae = minkowski(data['tsc'], data['esc'], 1) / n_samples
rmse = np.sqrt(sqeuclidean(data['tsc'], data['esc']) / n_samples)
mi = mi_disc(data['esc'], data['view'],
    [-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf], 2)
msc = [np.mean(data['esc'][data['view'] == view]) for view in xrange(2)]

#sys.stdout.write(infile + "\t")
sys.stdout.write(str(mae) + "\t")
sys.stdout.write(str(rmse) + "\t")
sys.stdout.write("\t".join([str(x) for x in mi]) + "\t")
sys.stdout.write("\t".join([str(x) for x in msc]) + "\n")
