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

import unittest

from mcnets import benchmark
from mcnets import models
from mcnets import testutils
from mcnets import utils


class TestBenchmark(testutils.StochasticTestCase):

  def GetBenchmarkScore(self, make_benchmark, make_chain):
    score = make_benchmark().Run(
        make_chain=lambda: make_chain(rng=self.rng),
        nchains=100,
        ntransitions=5000,
        burn=100)
    return score

  def assertFastChainWins(self, make_benchmark):
    slow_score = self.GetBenchmarkScore(
        make_benchmark,
        models.SlowTernaryMetropolisChain)
    fast_score = self.GetBenchmarkScore(
        make_benchmark,
        models.TernaryMetropolisChain)
    self.assertGreater(fast_score, slow_score)

  def testGewekeBenchmark(self):
    self.assertFastChainWins(benchmark.GewekeBenchmark)

  def testGelmanRubinBenchmark(self):
    self.assertFastChainWins(benchmark.GelmanRubinBenchmark)

  def testExpectationBenchmark(self):
    true_expectation = utils.Expectation(
        models.TernaryMetropolisChain.stationary_dist)
    expectation_benchmark = benchmark.GetExpectationBenchmark(true_expectation)
    self.assertFastChainWins(expectation_benchmark)


if __name__ == "__main__":
  unittest.main()
