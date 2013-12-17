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

"""Various utilities, mainly for handling randomness."""
from __future__ import division

import itertools
import math

import numpy as np


LOG_PROB_0 = float("-inf")
LOG_PROB_1 = 0.0
VERY_LARGE_NUMBER = 10**10


class SimpleNamespace(object):
  """See
  http://docs.python.org/3/library/types.html#types.SimpleNamespace
  """

  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  def __repr__(self):
    keys = sorted(self.__dict__)
    items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
    return "{}({})".format(type(self).__name__, ", ".join(items))


def Normalize(xs):
  """Rescales elements in xs to sum to 1."""
  z = sum(xs)
  assert z > 0, z
  return [x/z for x in xs]


def NormalizeDist(dic):
  """Rescales numeric values of dictionary to sum to 1.

  Args:
    dic: A dictionary with arbitrary keys and numeric values

  Returns:
    normalized_dic: dictionary with values rescaled to sum to 1.
  """
  z = sum(dic.values())
  assert z > 0, z
  return dict((v, p/z) for v, p in dic.items())


def Expectation(dist, f=None):
  """Returns the expected value of a distribution with numeric values."""
  if not f:
    f = lambda x: x
  expected_value = 0.0
  for value, prob in dist.items():
    expected_value += prob * f(value)
  return expected_value


class RandomState(np.random.RandomState):
  """Extends NumPy's RandomState with additional sampling functions."""

  def flip(self, prob):
    """Samples a weighted coin (Bernoulli)."""
    return self.rand() < prob

  def log_flip(self, log_prob):
    """Samples a weighted coin (Bernoulli) given log weight."""
    return self.rand() < math.exp(log_prob)

  def random_permutation(self, x):
    """Return shuffled copy of array, or shuffled range for int."""
    if isinstance(x, (int, np.integer)):
      arr = np.arange(x)
    else:
      arr = np.array(x)
    self.shuffle(arr)
    return arr

  def dict_to_sampler(self, dist):
    """Converts a discrete distribution (dict) to a sampling function."""
    values, weights = zip(*dist.items())
    return self.list_to_sampler(values, weights)

  def list_to_sampler(self, values, weights):
    """Converts lists of values and weights to a sampling function."""
    # values may be a numpy array, which can't be coerced to Boolean,
    # hence we convert to list in the assertion.
    assert list(values)
    assert len(values) == len(weights)
    probabilities = Normalize(weights)
    bins = np.add.accumulate(probabilities)
    def Sampler():
      index = np.digitize([self.rand()], bins)[0]
      return values[index]
    return Sampler


def GeometricProgression(v0, vn, n):
  """Returns a geometric sequence in interval [v0, vn] with n steps."""
  step = (vn / v0)**(1.0/(n - 1))
  return [v0 * (step ** k) for k in xrange(n)]


def Pairwise(iterable):
  """Returns an iterator over adjacent pairs of the original iterable."""
  a, b = itertools.tee(iterable)
  next(b, None)
  return itertools.izip(a, b)


def Mix(xs, ys, epsilon):
  """Convex combination of lists xs and ys: (1-epsilon)*xs + epsilon*ys."""
  return [x*(1-epsilon) + y*epsilon for x, y in zip(xs, ys)]


class SymmetricArray(np.ndarray):
  """A 2D numpy array that enforces that upper/lower triangle are symmetric."""

  def __setitem__(self, (i, j), value):
    np.ndarray.__setitem__(self, (i, j), value)
    np.ndarray.__setitem__(self, (j, i), value)


def Symmetrize(array):
  """Converts 2D numpy array to array class that enforces symmetry.

  This is intended for generating weighted adjacency matrices for
  undirected graphs from weighted adjacency matrices for directed
  graphs. Given a matrix, add up elements (i, j) and (j, i) (since
  they now refer to the same edge), subtract the diagonal (we don't
  want to double-count edge (i, i)) and convert to a symmetric array
  (to ensure that future assignments to (i, j) automatically update
  (j, i).).

  Args:
    array: a 2D square numpy array

  Returns:
    symmetric_a: an instance of SymmetricArray initialized with values of array.
  """
  symmetric_array = array + array.T - np.diag(array.diagonal())
  return symmetric_array.view(SymmetricArray)


def SymmetricOnes(n):
  """Returns a persistently symmetric square matrix with all entries set to 1.

  On assignment to (i, j), the returned matrix object automatically
  updates (j, i) as well.

  Args:
    n: the number of rows and columns of the square matrix.

  Returns:
    symmetric_array: a symmetric square matrix, with all entries set to 1.
  """
  return Symmetrize(np.tril(np.ones(shape=(n, n))))


def SymmetricZeros(n):
  """Returns a persistently symmetric square matrix with all entries set to 0.

  On assignment to (i, j), the returned matrix object automatically
  updates (j, i) as well.

  Args:
    n: the number of rows and columns of the square matrix.

  Returns:
    symmetric_array: a symmetric square matrix, with all entries set to 0.
  """
  return Symmetrize(np.zeros(shape=(n, n)))
