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

from __future__ import division


import unittest

from mcnets import stateless
from mcnets import testutils
from mcnets import utils


class TestStatelessNetworkChain(testutils.StochasticTestCase):

  def testBinaryNetwork(self):
    swap_graph = utils.SymmetricOnes(2)
    chain = stateless.StatelessNetworkChain(
        swap_graph, burn_roundtrips=10, rng=self.rng)
    chain.TransitionN(10000)
    stats = chain.statistics
    self.assertEqual(stats.transitions, 10000)
    self.assertEqual(stats.chain_scores[0], stats.chain_scores[1])
    self.assertEqual(stats.edge_scores[0, 1], stats.edge_scores[1, 0])
    self.assertGreater(stats.edge_scores[0, 1], 0.0)
    self.assertEqual(stats.edge_scores[0, 0], 0.0)
    self.assertEqual(stats.edge_scores[1, 1], 0.0)

  def testTernaryNetwork(self):
    swap_graph = utils.SymmetricOnes(3)
    swap_graph[0, 2] = 10
    chain = stateless.StatelessNetworkChain(
        swap_graph, burn_roundtrips=10, rng=self.rng)
    chain.TransitionN(10000)
    stats = chain.statistics
    self.assertEqual(stats.transitions, 10000)
    self.assertGreater(stats.chain_scores[0], stats.chain_scores[1])
    self.assertGreater(stats.chain_scores[2], stats.chain_scores[1])
    self.assertEqual(stats.edge_scores[0, 1], stats.edge_scores[1, 0])
    self.assertGreater(stats.edge_scores[0, 2], stats.edge_scores[0, 1])


if __name__ == "__main__":
  unittest.main()
