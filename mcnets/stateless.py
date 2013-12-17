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

"""Stateless chains for testing and simulation."""
from __future__ import division

from mcnets import mcmc
from mcnets import network
from mcnets import tempering


class StatelessChain(mcmc.MarkovChain):
  """A dummy Markov chain without state."""

  def InitializeState(self):
    pass

  def Transition(self):
    pass


class StatelessChainWithTemperature(StatelessChain):
  """A stateless chain with (unused) temperature parameter."""

  def __init__(self, temp, rng=None):
    self.SetTemperature(temp)
    super(StatelessChainWithTemperature, self).__init__(rng=rng)

  def SetTemperature(self, temp):
    self.temp = temp     # For compatibility with TemperedChain
    self.temps = [temp]  # For compatibility with NetworkChain


class StatelessAdaptiveTemperedChain(tempering.AdaptiveTemperedChain):
  """Tempered chain with stateless component chains."""

  def __init__(self, swap_probs, burn_roundtrips, rng=None):
    chains = [StatelessChainWithTemperature(temp, rng=rng)
              for temp in range(1, len(swap_probs)+2)]
    assert len(chains) == len(swap_probs) + 1
    self.swap_probs = swap_probs
    super(StatelessAdaptiveTemperedChain, self).__init__(
        chains, nsteps_per_sweep=0, nswaps_per_sweep=1,
        burn_roundtrips=burn_roundtrips, rng=rng)

  def BaseTransition(self):
    pass

  def SwapTransition(self, c1, c2, level):
    swap_accepted = self.rng.rand() < self.swap_probs[level]
    self.UpdateReplicaStatistics(swap_accepted, level)


class StatelessNetworkChain(network.NetworkChain):
  """Network chain with stateless component chains."""

  def __init__(self, swap_graph, burn_roundtrips, rng=None):
    chains = [StatelessChainWithTemperature(temp=0.5, rng=rng)
              for _ in range(len(swap_graph))]
    chains[0].SetTemperature(0.0)
    chains[-1].SetTemperature(1.0)
    super(StatelessNetworkChain, self).__init__(
        chains, nsteps_per_sweep=0, nswaps_per_sweep=1,
        burn_roundtrips=burn_roundtrips, swap_graph=swap_graph, rng=rng)

  def BaseTransition(self):
    pass

  def SwapTransition(self, chain1_index, chain2_index):
    swap_accepted = True
    self.UpdateReplicaStatistics(swap_accepted, chain1_index, chain2_index)
    return swap_accepted
