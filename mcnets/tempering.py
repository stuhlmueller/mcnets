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

"""Tempered Markov chains, with fixed and adaptively set temperatures.

TODO(stuhlmueller): Refactor TemperedChain and NetworkChain to share
more code (e.g., by building a ReplicaStatsMixin class).
"""
from __future__ import division

import numpy as np
from scipy import interpolate

from mcnets import mcmc
from mcnets import utils


class TemperedChain(mcmc.MetropolisHastingsChain):
  """A compound chain that interleaves elementary transitions with swaps."""

  def __init__(self, chains, nsteps_per_sweep, nswaps_per_sweep, rng=None):
    """Initializes tempered chain based on component chains.

    Args:
      chains: list of component chains, in increasing order of temperature
      nsteps_per_sweep: the number of transitions to apply to each
        component chain before we propose to swap neighboring pairs of
        chains.
      nswaps_per_sweep: the number of swaps to execute each time the
        network chain transition method is called.
      rng: a random stream (utils.RandomState).
    """
    assert len(chains) > 1, len(chains)
    self.CheckTemperatureOrder(chains)
    self.chains = chains
    self.nsteps_per_sweep = nsteps_per_sweep
    self.nswaps_per_sweep = nswaps_per_sweep
    super(TemperedChain, self).__init__(rng=rng)

  def InitializeState(self):
    """The chain doesn't keep state beyond the state of its component chains."""
    for chain in self.chains:
      chain.InitializeState()

  def InitializeStatistics(self):
    """Keeps track of statistics for each rank in the temperature ladder.

    Statistics:
      transitions: the number of transitions; each transition consists
        of len(self.chains)*nsteps_per_sweep base moves and
        len(self.chains)-1 swap moves.
      swaps_proposed: map chain index to number of swap proposals with
        neighbor at next-higher temperature
      swaps_accepted: map chain index to number of accepted swaps with
        neighbor at next-higher temperature
    """
    n = len(self.chains)
    self.statistics = utils.SimpleNamespace(
        transitions=0,
        swaps_proposed=np.zeros(n-1),
        swaps_accepted=np.zeros(n-1)
    )
    for chain in self.chains:
      chain.InitializeStatistics()

  def CheckTemperatureOrder(self, chains):
    """Verifies that chains are given in ascending order of temperatures."""
    for chain1, chain2 in utils.Pairwise(chains):
      assert hasattr(chain1, "temp")
      assert hasattr(chain2, "temp")
      assert chain1.temp < chain2.temp, (chain1.temp, chain2.temp)

  def SwapTransition(self, chain1, chain2, level):
    """Proposes to swap the states of two chains, accepts according to MH rule.

    Args:
      chain1: first chain to take part in swap
      chain2: second chain to take part in swap
      level: index of the lower-temperature chain

    Returns:
      swap_accepted: a Boolean indicating whether the swap was accepted
    """
    assert chain1 != chain2
    self.statistics.swaps_proposed[level] += 1
    logp_old = (chain1.LogProbability(chain1.state) +
                chain2.LogProbability(chain2.state))
    logp_new = (chain1.LogProbability(chain2.state) +
                chain2.LogProbability(chain1.state))
    log_acceptance_ratio = logp_new - logp_old
    swap_accepted = (log_acceptance_ratio >= utils.LOG_PROB_1 or
                     self.rng.log_flip(log_acceptance_ratio))
    if swap_accepted:
      self.statistics.swaps_accepted[level] += 1
      chain1.state, chain2.state = chain2.state, chain1.state
    return swap_accepted

  def BaseTransition(self):
    """Applies transition operator to component chains."""
    for chain in self.chains:
      chain.TransitionN(self.nsteps_per_sweep)

  def Transition(self):
    """Executes nsteps_per_sweep local transitions for each chain, then swaps.

    Swap strategy: Repeat nswaps_per_sweep times: pick an edge in the
    temperature ladder at random, propose to swap the chains connected
    by this edge.
    """
    self.BaseTransition()
    for _ in xrange(self.nswaps_per_sweep):
      level = self.rng.randint(len(self.chains)-1)
      self.SwapTransition(self.chains[level], self.chains[level+1], level)
    self.statistics.transitions += 1

  @property
  def state(self):
    """Only the state of the base chain is visible externally."""
    return self.chains[0].state

  @property
  def temps(self):
    """Returns the list of temperatures of the component chains (ascending)."""
    return [chain.temp for chain in self.chains]


def ComputeAdaptedTemperatures(temps, hist, epsilon=.001):
  """Given statistics about replica behavior, computes new temperatures.

  This function is based on the presentation of the Feedback-Optimized
  Parallel Tempering algorithm in [1]. The algorithm requires that the
  histogram describing the fraction of replicas flowing upwards (for
  each temperature) is monotonically decreasing in temperature. To
  ensure this, we mix in epsilon of a linearly decreasing histogram.

  [1] Robust Parameter Selection for Parallel Tempering
      Firas Hamze, Neil Dickson, Kamran Karimi (2010)
      http://arxiv.org/abs/1004.2840

  Args:
    temps: a list of temperatures (positive floating-point
      numbers, strictly increasing)
    hist: for each temperature, the fraction of replicas that are
      on the way up (decreasing, in [0, 1])
    epsilon: a number in [0, 1] that determines how strongly we mix a
      linearly decreasing function into hist (to ensure that it
      is strictly monotonically decreasing, and hence invertible)

  Returns:
    new_temps: an adapted list of temperatures
  """
  n = len(temps)
  assert n > 1
  assert len(hist) == n, (len(hist), n)
  for temp1, temp2 in utils.Pairwise(temps):
    assert temp1 > 0, temp1
    assert temp1 < temp2, (temp1, temp2)
  for p1, p2 in utils.Pairwise(hist):
    assert 0 <= p1 <= 1, p1
    assert p1 >= p2, (p1, p2)
  # Create a linearly spaced list of numbers in [0, 1] with n elements.
  # For example, for n=5, linear_hist = [1.0, 0.75, 0.5, 0.25, 0.0].
  linear_hist = [x/(n-1) for x in reversed(range(n))]
  stricly_monotonic_hist = utils.Mix(hist, linear_hist, epsilon)
  fraction_to_temp = interpolate.interp1d(
      list(reversed(stricly_monotonic_hist)),
      list(reversed(temps)))
  new_temps = [temps[0]]
  for i in reversed(range(2, n)):
    new_temps.append(float(fraction_to_temp((i-1)/(n-1))))
  new_temps.append(temps[-1])
  return new_temps


class AdaptiveTemperedChain(TemperedChain):
  """Adaptive tempering for Markov Chains based on [1] and [2].

  We want to maximize the rate of round trips that each replica
  (state) performs between the two extremal temperatures. We collect
  for each temperature statistics about how many replicas we have
  observed that have visited the lowest temperature most recently
  (replicas that are "on the way up"), and how many have visited the
  highest temperature more recently ("on the way down").

  These statistics allow us to calculate for each temperature the
  fraction of replicas which have visited one of the extremal
  temperatures most recently. Ideally, this fraction decreases
  linearly over the range of temperatures. Therefore, adaptation
  rearranges the temperatures based on the observed fractions such
  that we would have observed a linear decrease if we had observed the
  actual fractions together with the rearranged temperatures.

  To ensure that the empirical histogram of fractions is strictly
  monotonically decreasing as temperatures grow (a condition necessary
  for the algorithm), we mix in epsilon of a list of linearly
  decreasing fractions.

  [1] Feedback-optimized parallel tempering Monte Carlo
      Helmut G. Katzgraber, Simon Trebst, David A. Huse, Matthias Troyer (2006)
      http://arxiv.org/abs/cond-mat/0602085
  [2] Robust Parameter Selection for Parallel Tempering
      Firas Hamze, Neil Dickson, Kamran Karimi (2010)
      http://arxiv.org/abs/1004.2840
  """

  def __init__(self, chains, nsteps_per_sweep, nswaps_per_sweep,
               burn_roundtrips=0, rng=None):
    """Initializes tempered chain based on component chains.

    Args:
      chains: list of component chains, in increasing order of temperature
      nsteps_per_sweep: the number of transitions to apply to each
        component chain before we propose to swap neighboring pairs of
        chains.
      nswaps_per_sweep: the number of swaps to execute each time the
        network chain transition method is called.
      burn_roundtrips: the number of replica roundtrips required
        before we start updating statistics about replica direction
        averages
      rng: a random stream (utils.RandomState)
    """
    super(AdaptiveTemperedChain, self).__init__(
        chains, nsteps_per_sweep, nswaps_per_sweep, rng=rng)
    self.burn_roundtrips = burn_roundtrips

  def InitializeStatistics(self):
    """Initializes acceptance and replica behavior acceptance statisics.

    Statistics:
      transitions: the number of transitions; each transition consists
        of len(self.chains)*self.nsteps_per_sweep base moves and
        len(self.chains)-1 swap moves.
      swaps_proposed: maps chain index to number of swap proposals
        with next-higher neighbor
      swaps_accepted: maps chain index to number of accepted swaps with
        next-higher neighbor
      replica: a record with statistics that describe the behavior of
        replica, i.e., the movement of states across different
        temperatures (see InitializeReplicaStatistics).
    """
    n = len(self.chains)
    self.statistics = utils.SimpleNamespace(
        transitions=0,
        swaps_proposed=np.zeros(n-1),
        swaps_accepted=np.zeros(n-1),
        replica=None)
    self.InitializeReplicaStatistics()
    for chain in self.chains:
      chain.InitializeStatistics()

  def InitializeReplicaStatistics(self):
    """Initializes replica behavior statistics.

    Statistics:
      ids: maps chain index to id of replica currently located at
        chain
      directions: maps replica id to {-1, 0, 1}, depending on whether
        replica last visited bottom-most chain (1, "upwards"), top
        chain (-1, "downwards"), or neither (0).
      tracker: maps chain index to a list that stores for every
        replica the direction it has been observed moving in most
        recently by chain
      hist: for each chain, cumulative moving average of fraction of
        replica observed moving upwards
      hist_n: for each chain, the number of observations that made it
        into the moving average
      roundtrips: the total number of replica that made it all the way
        from the top to the bottom
    """
    n = len(self.chains)
    replica_stats = utils.SimpleNamespace(
        ids=range(n),
        directions=np.zeros(n),
        tracker=[[0]*n for _ in range(n)],
        hist=np.zeros(n),
        hist_n=np.zeros(n),
        roundtrips=0)
    replica_stats.directions[0] = 1
    replica_stats.directions[-1] = -1
    replica_stats.hist[0] = 1
    self.statistics.replica = replica_stats

  def ResetReplicaStatistics(self):
    """Resets replica statistics to initial state."""
    self.InitializeReplicaStatistics()

  def UpdateReplicaStatistics(self, swap_accepted, level):
    """Updates information about replica behavior after a swap.

    Args:
      swap_accepted: a Boolean indicating whether the swap was accepted.
      level: the index of the swap chain with lower temperature
    """
    if not swap_accepted:
      return
    r = self.statistics.replica
    n = len(self.chains)
    # Swap the replica ids.
    r.ids[level], r.ids[level+1] = r.ids[level+1], r.ids[level]
    # Update roundtrips and directions of the replicas at top/bottom chains.
    if r.directions[r.ids[0]] == -1:
      r.roundtrips += 1
    r.directions[r.ids[0]] = 1
    r.directions[r.ids[-1]] = -1
    # Update tracker: store direction of new replicas at level, level+1.
    for k in (level, level+1):
      r.tracker[k][r.ids[k]] = r.directions[r.ids[k]]
    # Update moving average of replica directions (fraction moving up).
    for i in range(n):
      if self.statistics.replica.roundtrips > self.burn_roundtrips:
        n_up = r.tracker[i].count(1)
        n_down = r.tracker[i].count(-1)
        assert 0 < n_up + n_down <= n
        p = n_up / (n_up + n_down)
        r.hist[i] += (p - r.hist[i]) / (r.hist_n[i] + 1)
        r.hist_n[i] += 1

  def TransitionsPerRoundtrip(self, stats=None):
    """Returns number of transitions necessary for a single roundtrip.

    A single transition encompasses len(self.chains) *
    self.nsteps_per_sweep base steps and self.nswaps_per_sweep swaps.

    Args:
      stats: statistics of a NetworkChain.

    Returns:
      score: average number of transitions per roundtrip.
    """
    stats = stats if stats else self.statistics
    if not stats.replica.roundtrips:
      return float("inf")
    return stats.transitions / stats.replica.roundtrips

  def SwapTransition(self, chain1, chain2, level):
    """Proposes to swap states of two chains, accept according to MH rule."""
    swap_accepted = super(AdaptiveTemperedChain, self).SwapTransition(
        chain1, chain2, level)
    self.UpdateReplicaStatistics(swap_accepted, level)

  def AdaptTemperatures(self):
    """Computes and sets new temperatures based on replica statistics."""
    new_temps = ComputeAdaptedTemperatures(
        self.temps, self.statistics.replica.hist)
    for chain, new_temp in zip(self.chains, new_temps):
      chain.temp = new_temp
