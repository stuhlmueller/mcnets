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

import collections
import copy
import math

import unittest

from mcnets import models
from mcnets import network
from mcnets import testutils
from mcnets import utils


class ReparameterizedIsing(models.IsingChain):
  """Reparameterize Ising to use temperatures in unit interval.

  In contrast to the tempering algorithms, the network scheme assumes
  that temperatures are in [0, 1], and it allows more than one
  temperature dimension. In order to use the Ising model for testing
  the network scheme, we wrap the model in a class that redirects
  access to the "temps" parameter (used by the network scheme) to the
  "temp" parameter (used by the tempering scheme), rescaling
  reads/writes to/from the unit interval.
  """

  def __init__(self, size, unit_temp, max_temp, rng):
    assert 0 <= unit_temp <= 1, unit_temp
    assert max_temp > 0
    self.max_temp = max_temp
    temp = self.UnitTempToTemp(unit_temp)
    super(ReparameterizedIsing, self).__init__(size, temp=temp, rng=rng)

  def UnitTempToTemp(self, unit_temp):
    return self.max_temp ** unit_temp

  def TempToUnitTemp(self, temp):
    return math.log(temp) / math.log(self.max_temp)

  def __setattr__(self, name, val):
    if name == "temps":
      assert len(val) == 1
      self.temp = self.UnitTempToTemp(val[0])
    else:
      return super(ReparameterizedIsing, self).__setattr__(name, val)

  def __getattr__(self, name):
    if name == "temps":
      return [self.TempToUnitTemp(self.temp)]
    else:
      raise AttributeError(name)


class IsingNetworkTestCase(testutils.StochasticTestCase):
  """A test case with methods for setting up Ising chain networks."""

  def GetIsingChain(self, unit_temp):
    """Return a single Ising chain at the given temperature."""
    return ReparameterizedIsing(
        size=self.ising_size,
        unit_temp=unit_temp,
        max_temp=self.max_temp,
        rng=self.rng)

  def GetNetworkChain(self, network_chain_class):
    self.ising_size = 4
    self.unit_temps = [0.0, 0.25, 0.5, 0.75, 1.0]
    self.max_temp = 16
    chains = [self.GetIsingChain(unit_temp) for unit_temp in self.unit_temps]
    network_chain = network_chain_class(
        chains=chains,
        nsteps_per_sweep=100,
        nswaps_per_sweep=10,
        burn_roundtrips=0,
        rng=self.rng)
    return network_chain


class TestNetworkChain(IsingNetworkTestCase):

  def setUp(self):
    self.network_chain = self.GetNetworkChain(network.NetworkChain)

  def testTransition(self):
    """Check that elementary and swap transitions work."""
    self.network_chain.TransitionN(10)

  def testSwapIndexSampler(self):
    """Check that distribution of swaps is uniform and without self-swaps."""
    sampler = self.network_chain.GetSwapIndexSampler()
    counts = collections.defaultdict(lambda: 0)
    for _ in xrange(100000):
      (i, j) = sampler()
      counts[(i, j)] += 1
    for ((i, j), count) in counts.items():
      self.assertAlmostEqual(count, 10000, delta=300)
      self.assertNotEqual(i, j)

  def testStatistics(self):
    """Check that collected statistics are correct."""
    stats = self.network_chain.statistics
    r = stats.replica
    n = len(self.network_chain.chains)
    # Check initial statistics
    self.assertEqual(stats.transitions, 0)
    for i in range(n):
      self.assertEqual(stats.chain_scores[i], 0)
      for j in range(n):
        self.assertEqual(stats.edge_scores[i, j], 0)
      self.assertEqual(r.ids[i], i)
      self.assertEqual(r.directions[i], 0)
      self.assertEqual(len(r.traversed[i]), 0)
      self.assertEqual(r.roundtrips, 0)
    # Execute transitions
    self.network_chain.TransitionN(200)
    # Check post-transition statistics
    self.assertEqual(stats.transitions, 200)
    for i in range(n):
      self.assertGreater(stats.chain_scores[i], 0)
      for j in range(n):
        if i == j:
          self.assertEqual(stats.edge_scores[i, j], 0)
        elif abs(i - j) == 1:
          # We can be fairly sure that edges between nearby
          # temperatures will be traversed.
          self.assertGreater(stats.edge_scores[i, j], 0)
      self.assertNotEqual(r.directions[i], 0)
      if i > 0:
        self.assertGreater(len(r.traversed[r.ids[i]]), 0)
      self.assertGreater(r.roundtrips, 0)


class TestAdaptiveNetworkChain(IsingNetworkTestCase):

  def setUp(self):
    self.network_chain = self.GetNetworkChain(network.AdaptiveNetworkChain)

  def testParameters(self):
    """Check that getting/setting parameters works."""
    params = self.network_chain.GetParameters()
    self.network_chain.nswaps_per_sweep = 30
    self.assertNotEqual(params.nswaps_per_sweep,
                        self.network_chain.nswaps_per_sweep)
    self.network_chain.SetParameters(params)
    self.assertEqual(params.nswaps_per_sweep,
                     self.network_chain.nswaps_per_sweep)

  def testNoisifyTemperatures(self):
    """Check that new temperatures are in [0, 1], close to to old temps."""
    temps = [0.0, 0.3, 0.6, 1.0]
    new_temps = self.network_chain.NoisifyTemperatures(
        temps, pseudocounts=10000)
    self.assertGreater(new_temps[0], 0.0)
    self.assertLess(new_temps[-1], 1.0)
    for temp, new_temp in zip(temps, new_temps):
      self.assertAlmostEqual(temp, new_temp, delta=.01)

  def testAdaptTemperatures(self):
    """Check that worst chain temps get replaced with noisy copy of best."""
    chain = self.network_chain
    chain.TransitionN(5)
    chain.statistics.chain_scores = [i for i in xrange(len(chain.chains))]
    chain.AdaptTemperatures()
    self.assertAlmostEqual(
        chain.chains[0].temps[0], chain.chains[-1].temps[0], delta=.03)
    self.assertAlmostEqual(chain.chains[0].temps[0], 1.0, delta=.03)

  def testAdaptSwapGraphUnchanged(self):
    """If baseline and empirical stats equal, check that graph is unchanged."""
    chain = self.network_chain
    swap_graph_pre = copy.copy(chain.swap_graph)
    chain.TransitionForRoundtrips(5)
    chain.AdaptSwapGraph(chain.statistics)
    self.assertArraysEqual(swap_graph_pre, chain.swap_graph)

  def testAdaptSwapGraphChanged(self):
    """If baseline stats differ, check that graph is changed correctly."""
    chain = self.network_chain
    chain.TransitionForRoundtrips(5)
    chain.statistics.edge_scores = utils.SymmetricOnes(len(chain.chains))
    baseline_stats = copy.deepcopy(chain.statistics)
    baseline_stats.edge_scores[1, 2] = 5
    baseline_stats.edge_scores[0, 3] = 0.1
    chain.AdaptSwapGraph(baseline_stats)
    self.assertAlmostEqual(chain.swap_graph[0, 3], 10.0)
    self.assertAlmostEqual(chain.swap_graph[1, 3], 1.0)
    self.assertAlmostEqual(chain.swap_graph[1, 2], 0.2)

  def testAdaptation(self):
    """Check that adaptation decreases number of transitions per roundtrip."""
    chain = self.network_chain
    chain.Adapt(
        generations=5,
        roundtrips_per_generation=20,
        adapt_temps_every=5)
    initial_stats = chain.GetRevision(0).statistics
    final_stats = chain.PreviousRevision().statistics
    initial_cost = chain.TransitionsPerRoundtrip(initial_stats)
    final_cost = chain.TransitionsPerRoundtrip(final_stats)
    self.assertLess(final_cost, initial_cost)


if __name__ == "__main__":
  unittest.main()
