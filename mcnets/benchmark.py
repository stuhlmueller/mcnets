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

"""Benchmarks for Markov chain algorithms.

This file contains benchmarks used to diagnose and compare the
convergence of Markov Chain Monte Carlo algorithms. Benchmarks return
scores in [0, 1]. These scores are not comparable across benchmarks;
the only meaningful use is for relative ordering within a benchmark
class.
"""
import math

import numpy as np
from pymc import diagnostics
from scipy import stats

from mcnets import utils


class Benchmark(object):
  """Abstract class for Markov chain benchmarks."""

  def __init__(self, transform_state=None, error_percentile=90):
    """Initializes Benchmark with state transformer and error percentile.

    Args:
      transform_state: A function that maps a Markov chain state to a
                        value that can be handled by TraceError.
      error_percentile: Percentile to use when aggregating errors across traces.
    """
    self.traces = []
    self.current_trace = None
    self.transform_state = transform_state
    self.error_percentile = error_percentile

  def TraceError(self, trace):
    """Compute error in [0, inf] for a single trace."""
    raise NotImplementedError()

  def TotalError(self):
    """Compute error in [0, inf] for a list of traces."""
    assert self.traces
    trace_errors = [self.TraceError(trace) for trace in self.traces]
    total_error = stats.scoreatpercentile(trace_errors, self.error_percentile)
    return total_error

  def Score(self):
    """Returns score in [0, 1] for stored traces (1 is best)."""
    error = self.TotalError()
    assert error >= 0
    return math.exp(-error)

  def TransformState(self, state):
    """Transform Markov chain state into a form suitable for TraceError."""
    if self.transform_state:
      return self.transform_state(state)
    else:
      return state

  def StartTrace(self):
    """Adds a new empty list (trace) to the list of traces."""
    trace = []
    self.traces.append(trace)
    self.current_trace = trace

  def RecordState(self, state):
    """Store transformed version of state in current trace."""
    self.current_trace.append(self.TransformState(state))

  def Run(self, make_chain, nchains, ntransitions, burn):
    """Runs a single benchmark, returning the final score."""
    scores = RunBenchmarks(make_chain, [self], nchains, ntransitions, burn)
    return scores[self.name]

  @property
  def name(self):
    return self.__class__.name


def RunBenchmarks(make_chain, benchmarks, nchains, ntransitions, burn):
  """Runs a set of benchmarks, returning a dictionary with scores.

  Args:
    make_chain: a thunk that returns a Markov chain
    benchmarks: a list of benchmarks
    nchains: the number of Markov chains to generate and run
    ntransitions: the number of steps for each Markov chain (after burn-in)
    burn: the number of initial samples to discard as burn-in

  Returns:
    scores: a dictionary mapping benchmark name to score
  """
  for _ in xrange(nchains):
    chain = make_chain()
    for benchmark in benchmarks:
      benchmark.StartTrace()
    for step in xrange(ntransitions + burn):
      chain.Transition()
      if step > burn:
        for benchmark in benchmarks:
          benchmark.RecordState(chain.state)
  scores = dict((benchmark.name, benchmark.Score())
                for benchmark in benchmarks)
  return scores


class GelmanRubinBenchmark(Benchmark):
  """Compares within-chain and between-chain variances.

  The statistic r_hat should be <= 1 for convergent Markov chains.
  Hence, the error we compute is the distance by which r_hat exceeds 1.
  """
  name = "Gelman-Rubin"

  def TotalError(self):
    """Compute Gelman-Rubin error for a set of traces."""
    assert self.traces
    try:
      # r_hat is the "potential scale reduction", a convergence
      # diagnostic computed by comparing between-chain and
      # within-chain variance.
      r_hat = diagnostics.gelman_rubin(self.traces)
    except FloatingPointError:
      # This happens when the within-chain standard deviation is
      # 0. The chain perfectly correlates with itself, implying
      # complete failure to converge (if the chain has more than one
      # state).
      return utils.VERY_LARGE_NUMBER
    return max(0, r_hat - 1)


class ExpectationBenchmark(Benchmark):
  """Compares sample expectations to true expectations.

  The error we compute is the absolute distance between the true and
  sample expectations.
  """
  name = "Expectation"

  def __init__(self, true_expectation, **kwargs):
    super(ExpectationBenchmark, self).__init__(**kwargs)
    self.true_expectation = true_expectation

  def TraceError(self, trace):
    """Compute absolute distance between true and sample means."""
    return abs(np.mean(trace) - self.true_expectation)


def GetExpectationBenchmark(true_expectation, **kwargs):
  """Return an ExpectationBenchmark class for a particular true expectation."""

  class CurriedExpectationBenchmark(ExpectationBenchmark):
    def __init__(self):
      super(CurriedExpectationBenchmark, self).__init__(
          true_expectation, **kwargs)
  return CurriedExpectationBenchmark


class GewekeBenchmark(Benchmark):
  """Compares the mean of first % of samples with mean of last %.

  The z-scores returned by Geweke should be in [-1, 1] for convergent
  chains. Hence, the error we compute is the maximal distance of any
  z-score from this interval.
  """
  name = "Geweke"

  def TraceError(self, trace):
    """Compute summary statistic of Geweke z-scores for a trace."""
    try:
      geweke_scores = diagnostics.geweke(np.array(trace))
    except FloatingPointError:
      # This happens when the standard deviation is 0 for a trace
      # slice. The chain perfectly correlates with itself, implying
      # complete failure to converge (if the chain has more than one
      # state).
      return utils.VERY_LARGE_NUMBER
    z_scores = [z_score for _, z_score in geweke_scores]
    max_score = max(z_scores + [1])
    min_score = min(z_scores + [-1])
    return max(max_score-1, 1-min_score)
