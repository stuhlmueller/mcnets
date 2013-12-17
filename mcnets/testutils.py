## Copyright 2013 Google Inc. All Rights Reserved.
##
## Licensed under the Apache License, Version 2.0 (the )
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an AS IS BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

"""Various utilities for testing stochastic code."""
from __future__ import division

import collections

import numpy as np
from scipy import stats

import unittest

from mcnets import utils


def TTestGreaterThan(a, b, alpha=0.05):
  """Performs one-sided, independent T-test with null-hypothesis a <= b.

  The null-hypothesis is that a has a mean less than or equal to
  b. TTestGreaterThan returns a Boolean indicating whether the null
  hypothesis is rejected. In other words, if TTestGreaterThan returns
  true, a is likely to have a greater mean than b.

  Arguments:
    a: array-like
    b: array-like
    alpha: significance threshold in (0, 1)

  Returns:
    null_rejected: Boolean indicating whether the null hypothesis is rejected
  """
  t, prob = stats.ttest_ind(a, b)
  return prob/2 < alpha and t > 0


class StochasticTestCase(unittest.TestCase):
  """A test case that provides methods for checking distributions."""

  def __init__(self, *args, **kwargs):
    super(StochasticTestCase, self).__init__(*args, **kwargs)
    # If no random state is given, set it deterministically to make
    # test reproducible.
    self.rng = kwargs.get("rng") or utils.RandomState(0)

  def assertArraysEqual(self, array1, array2):
    for (x, y) in zip(np.nditer(array1), np.nditer(array2)):
      self.assertEqual(x, y)

  def assertAllAlmostEqual(self, xs, margin):
    mean = np.mean(xs)
    for x in xs:
      self.assertAlmostEqual(x, mean, delta=margin)

  def assertNotAllAlmostEqual(self, xs, margin):
    mean = np.mean(xs)
    all_almost_equal = True
    for x in xs:
      if abs(x - mean) > margin:
        all_almost_equal = False
    self.assertFalse(all_almost_equal)

  def assertInInterval(self, x_est, x_true, margin):
    self.assertLess(x_true - margin, x_est)
    self.assertLess(x_est, x_true + margin)

  def assertDistributionAlmostEqual(self, true_dist, est_dist, margin):
    true_dist = utils.NormalizeDist(true_dist)
    est_dist = utils.NormalizeDist(est_dist)
    for key, p_true in true_dist.items():
      p_est = est_dist.get(key, 0.0)
      self.assertInInterval(p_est, p_true, margin)

  def assertHasDistribution(self, thunk, dist, nsamples, margin):
    sample_dist = collections.defaultdict(lambda: 0)
    for _ in range(nsamples):
      sampled_value = thunk()
      sample_dist[sampled_value] += 1
    self.assertDistributionAlmostEqual(dist, sample_dist, margin)

  def assertHasMean(self, thunk, true_mean, nsamples, margin):
    sample_mean = 0.0
    for i in range(1, nsamples):
      sampled_value = thunk()
      sample_mean += (sampled_value - sample_mean) / i
    self.assertLess(true_mean - margin, sample_mean)
    self.assertLess(sample_mean, true_mean + margin)

  def assertSignificantlyGreater(self, a, b, alpha=0.05):
    """Perform t-test to check that sample a is significantly greater than b.

    Args:
      a: array-like
      b: array-like
      alpha: significance threshold for t-test
    """
    is_significantly_greater = TTestGreaterThan(a, b, alpha=alpha)
    self.assertTrue(is_significantly_greater,
                    msg=("%s not significantly greater than %s "
                         "at threshold alpha={:.5}.".format(a, b, alpha)))

  def assertNotSignificantlyGreater(self, a, b, alpha=0.05):
    """Perform t-test to check that a is not significantly greater than b.

    Args:
      a: array-like
      b: array-like
      alpha: significance threshold for t-test
    """
    is_significantly_greater = TTestGreaterThan(a, b, alpha=alpha)
    self.assertFalse(is_significantly_greater,
                     msg=("%s significantly greater than %s "
                          "at threshold alpha={:.5}.".format(a, b, alpha)))
