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

from scipy.spatial import distance

import unittest

from mcnets import benchmark
from mcnets import models
from mcnets import stateless
from mcnets import tempering
from mcnets import testutils
from mcnets import utils


class TestAdaptTemperatures(unittest.TestCase):
  """Test the temperature adaptation function in isolation."""

  def setUp(self):
    self.temps = range(1, 11)

  def assertTemperaturesConsistent(self, old_temps, new_temps):
    """Check relations that must hold between old and new temperatures.

    Args:
      old_temps: temperatures before adaptation.
      new_temps: temperatures after adaptation.

    Check that the number of temperatures and the min/max are
    unchanged, and that temperatures are in strict ascending order.
    """
    self.assertEqual(len(old_temps), len(new_temps))
    self.assertAlmostEqual(old_temps[0], new_temps[0])
    self.assertAlmostEqual(old_temps[-1], new_temps[-1])
    for temps in (old_temps, new_temps):
      for t1, t2 in utils.Pairwise(temps):
        self.assertLess(t1, t2)

  def GetReplicaHist(self, f):
    """Artifically create a histogram of observed replica directions.

    Args:
      f: function applied to linear hist entries to create the final histogram

    Returns:
      hist: list with (artificially created) fractions of replica moving upwards
    """
    n = len(self.temps)
    return [1-f((x-1)/(n-1)) for x in range(1, n+1)]

  def testTemperatureUpdateLinear(self):
    """Test that linear replica histogram results in (almost) no change."""
    hist = self.GetReplicaHist(lambda x: x)
    new_temps = tempering.ComputeAdaptedTemperatures(self.temps, hist)
    self.assertTemperaturesConsistent(self.temps, new_temps)
    for temp, new_temp in zip(self.temps, new_temps):
      self.assertAlmostEqual(temp, new_temp, delta=.0001)

  def testTemperatureUpdateConcave(self):
    """Test that concave replica histogram shifts temps upwards."""
    hist = self.GetReplicaHist(lambda x: x**4)
    new_temps = tempering.ComputeAdaptedTemperatures(self.temps, hist)
    self.assertTemperaturesConsistent(self.temps, new_temps)
    for temp, new_temp in zip(self.temps[1:-1], new_temps[1:-1]):
      self.assertLess(temp, new_temp)

  def testTemperatureUpdateConvex(self):
    """Test that convex replica histogram shifts temps downwards."""
    hist = self.GetReplicaHist(lambda x: x**(1/4.0))
    new_temps = tempering.ComputeAdaptedTemperatures(self.temps, hist)
    self.assertTemperaturesConsistent(self.temps, new_temps)
    for temp, new_temp in zip(self.temps[1:-1], new_temps[1:-1]):
      self.assertGreater(temp, new_temp)


class TestStatelessAdaptiveChain(testutils.StochasticTestCase):

  def setUp(self):
    self.nchains = 10
    self.swap_probs = [0.5] * (self.nchains - 1)
    self.ntransitions = 20000
    self.burn_roundtrips = 50

  def testHistAndTemps(self):
    chain = stateless.StatelessAdaptiveTemperedChain(
        self.swap_probs, self.burn_roundtrips, self.rng)
    chain.nswaps_per_sweep = 30
    pre_adaptation_temps = chain.temps
    chain.TransitionN(self.ntransitions)
    chain.AdaptTemperatures()
    chain.Reset()
    post_adaptation_temps = chain.temps
    # Verify that the temperatures remain unchanged for uniform swap probs.
    for pre, post in zip(pre_adaptation_temps, post_adaptation_temps):
      self.assertAlmostEqual(pre, post, delta=.2)
    # Verify that the histogram of replica directions is correct.
    chain.TransitionN(self.ntransitions)
    empirical_hist = chain.statistics.replica.hist
    true_hist = [1-x/(self.nchains-1) for x in range(self.nchains)]
    for empirical_p, true_p in zip(empirical_hist, true_hist):
      self.assertAlmostEqual(empirical_p, true_p, delta=.1)


class IsingTestCase(testutils.StochasticTestCase):
  """A base class for tempering test cases using the Ising model."""

  def setUp(self):
    raise NotImplementedError("IsingTestCase.setUp: set ising_size and temps!")

  def GetBenchmarks(self):
    """Return a list of freshly instantiated benchmarks for the Ising model.

    We arbitrarily analyze the dynamics of the Boolean variable at
    x=1, y=1 in the Ising state.

    Returns:
      benchmarks: a list of benchmarks
    """
    get_variable = lambda state: state[1][1]
    benchmarks = [
        benchmark.GelmanRubinBenchmark(
            transform_state=get_variable),
        benchmark.ExpectationBenchmark(
            true_expectation=0.0,
            transform_state=get_variable),
        benchmark.GewekeBenchmark(
            transform_state=get_variable)
    ]
    return benchmarks

  def GetIsingChain(self, temp):
    """Return a single Ising chain at the given temperature."""
    return models.IsingChain(size=self.ising_size, temp=temp, rng=self.rng)

  def GetTemperedIsingChain(self, nsteps_per_sweep):
    """Return a compound chain, composed of Ising chains at different temps."""
    chains = [self.GetIsingChain(temp) for temp in self.temps]
    tempered_chain = tempering.TemperedChain(
        chains=chains,
        nsteps_per_sweep=nsteps_per_sweep,
        nswaps_per_sweep=len(chains),
        rng=self.rng)
    return tempered_chain

  def GetAdaptiveIsingChain(self, nsteps_per_sweep, nswaps_per_sweep,
                            burn_roundtrips):
    """Return a (composite) chain that supports adaptation."""
    chains = [models.IsingChain(size=self.ising_size, temp=temp, rng=self.rng)
              for temp in self.temps]
    tempered_chain = tempering.AdaptiveTemperedChain(
        chains=chains,
        nsteps_per_sweep=nsteps_per_sweep,
        nswaps_per_sweep=nswaps_per_sweep,
        burn_roundtrips=burn_roundtrips,
        rng=self.rng)
    return tempered_chain

  def RunForRoundtrips(self, chain, nroundtrips, verbose=False):
    """Run chain until a given number of replica round trips is reached."""
    i = 0
    while chain.statistics.replica.roundtrips < nroundtrips:
      i += 1
      chain.Transition()
      if verbose and i % 10 == 0:
        print ("%i: %i, %s" %
               (chain.statistics.transitions,
                chain.statistics.replica.roundtrips,
                chain.statistics.replica.hist))

  def GetAdaptedIsingChain(self, nsteps_per_sweep, nswaps_per_sweep,
                           burn_roundtrips, adaptation_roundtrips,
                           adaptation_repeats):
    """Return a pre-adapted chain, composed of chains at different temps."""
    tempered_chain = self.GetAdaptiveIsingChain(
        nsteps_per_sweep, nswaps_per_sweep, burn_roundtrips)
    for _ in range(adaptation_repeats):
      self.RunForRoundtrips(tempered_chain, adaptation_roundtrips)
      tempered_chain.AdaptTemperatures()
      tempered_chain.Reset()
    return tempered_chain


class TestTemperingMCMC(IsingTestCase):
  """Test tempered MCMC without adaptation."""

  def setUp(self):
    self.ising_size = 3
    self.temps = utils.GeometricProgression(1.0, 20.0, 10)
    self.nsteps_per_sweep = 20
    self.nchains = 3
    self.ntransitions = 100
    self.burn = 10

  def testIsingPhaseTransition(self):
    """Verify acceptance prob dip at phase transition for geometric temps."""
    chain = self.GetTemperedIsingChain(self.nsteps_per_sweep)
    chain.TransitionN(self.ntransitions)
    swaps = chain.statistics.swaps_accepted
    self.assertLess(swaps[1], swaps[0])
    self.assertLess(swaps[1], swaps[-1])

  def testIsingBenchmark(self):
    """Assert that tempered convergence beats untempered convergence."""
    untempered_scores = benchmark.RunBenchmarks(
        make_chain=lambda: self.GetIsingChain(temp=1.0),
        benchmarks=self.GetBenchmarks(),
        nchains=self.nchains,
        ntransitions=self.ntransitions,
        burn=self.burn)
    tempered_scores = benchmark.RunBenchmarks(
        make_chain=lambda: self.GetTemperedIsingChain(self.nsteps_per_sweep),
        benchmarks=self.GetBenchmarks(),
        nchains=self.nchains,
        ntransitions=self.ntransitions * self.nsteps_per_sweep,
        burn=self.burn * self.nsteps_per_sweep)
    self.assertEqual(set(tempered_scores.keys()), set(untempered_scores.keys()))
    for name in tempered_scores:
      self.assertGreater(tempered_scores[name], untempered_scores[name])


class TestAdaptiveTemperingMCMC(IsingTestCase):
  """Test tempered MCMC with adaptation.

  We use linearly instead of geometrically spaced temperatures here in
  order to exaggerate the effect of adaptation. While adaptation
  improves performance even when temperatures are chosen
  geometrically, the effect is smaller and hence harder to detect in a
  short test case.

  TODO(stuhlmueller): Add reliable test using benchmarks.
  """

  def setUp(self):
    self.ising_size = 3
    self.temps = [2*i+1 for i in range(11)]
    self.nsteps_per_sweep = 20
    self.nswaps_per_sweep = 20
    self.nrepeats = 2
    self.nchains = 3
    self.burn = 10
    self.adaptation_roundtrips = 50
    self.burn_roundtrips = 10
    self.adaptation_repeats = 3

  def testIsingAcceptanceWithoutAdaptation(self):
    """Verify high variance in acceptance probs without adaptation."""
    ntransitions = 500
    chain = self.GetAdaptiveIsingChain(
        self.nsteps_per_sweep, self.nswaps_per_sweep, self.burn_roundtrips)
    chain.TransitionN(ntransitions)
    swaps = chain.statistics.swaps_accepted
    self.assertNotAllAlmostEqual(swaps, margin=150)

  def testIsingAcceptanceWithAdaptation(self):
    """Verify no acceptance prob dip at phase transition for adapted temps."""
    ntransitions = 1000
    chain = self.GetAdaptedIsingChain(
        self.nsteps_per_sweep, self.nswaps_per_sweep, self.burn_roundtrips,
        self.adaptation_roundtrips, self.adaptation_repeats)
    chain.TransitionN(ntransitions)
    swaps = chain.statistics.swaps_accepted
    # TODO(stuhlmueller): Figure out principled choice for margin.
    self.assertAllAlmostEqual(swaps, margin=250)

  def testReplicaHistAdaptation(self):
    """Verify that adaptation leads to linear replica directions."""
    chain = self.GetAdaptiveIsingChain(
        self.nsteps_per_sweep, self.nswaps_per_sweep, self.burn_roundtrips)
    ntemps = len(self.temps)
    linear_hist = [(ntemps/(ntemps-1))*(x/10) for x in reversed(range(ntemps))]
    # Run chain with linearly spaced temperatures.
    self.RunForRoundtrips(chain, self.adaptation_roundtrips)
    pre_ntransitions = chain.statistics.transitions
    pre_hist = chain.statistics.replica.hist
    # Verify that the histogram of replica directions is far from linear.
    self.assertGreater(distance.cityblock(linear_hist, pre_hist), 1.0)
    # Adapt temperatures.
    chain.AdaptTemperatures()
    chain.Reset()
    # Run chain with new temperatures.
    self.RunForRoundtrips(chain, self.adaptation_roundtrips)
    post_ntransitions = chain.statistics.transitions
    post_hist = chain.statistics.replica.hist
    # Verify that the histogram of replica directions is close to linear.
    self.assertLess(distance.cityblock(linear_hist, post_hist), 1.0)
    # Verify that the number of transitions necessary to reach the
    # same number of roundtrips is shorter after adaptation.
    self.assertLess(post_ntransitions, pre_ntransitions)


if __name__ == "__main__":
  unittest.main()
