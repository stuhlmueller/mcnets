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

"""Simple Markov chains for tests and benchmarks."""
from __future__ import division

import math

import numpy as np

from mcnets import mcmc
from mcnets import utils


class BinaryMarkovChain(mcmc.MarkovChain):
  """A 2-state Markov chain."""

  stationary_dist = {
      True: 3/5,
      False: 2/5
  }

  def InitializeState(self):
    self.state = True

  def Transition(self):
    if self.state:
      self.state = self.rng.flip(.8)
    else:
      self.state = self.rng.flip(.3)
    self.statistics.transitions += 1


class ParameterizedBinaryMarkovChain(mcmc.MarkovChain, mcmc.HistoryMixin):
  """A 2-state Markov chain."""

  def __init__(self, p_flip):
    self.p_flip = p_flip
    super(ParameterizedBinaryMarkovChain, self).__init__()

  def InitializeState(self):
    self.state = True

  def Transition(self):
    if self.state:
      self.state = self.rng.flip(1 - self.p_flip)
    else:
      self.state = self.rng.flip(self.p_flip)
    self.statistics.transitions += 1

  def GetParameters(self):
    return utils.SimpleNamespace(p_flip=self.p_flip)

  def SetParameters(self, params):
    self.p_flip = params.p_flip


class TernaryMetropolisChain(mcmc.MetropolisHastingsChain):
  """A 3-state Markov chain with MH proposal function."""

  stationary_dist = {
      0: 0.2,
      1: 0.3,
      2: 0.5
  }

  state_dist = {
      0: 0.2,
      1: 0.3,
      2: 0.5
  }

  transition_dists = {
      0: {0: 0.1,
          1: 0.2,
          2: 0.7},
      1: {0: 0.2,
          1: 0.2,
          2: 0.6},
      2: {0: 0.3,
          1: 0.4,
          2: 0.3}
  }

  def __init__(self, rng=None):
    super(TernaryMetropolisChain, self).__init__(rng=rng)
    self.transition_fns = {}
    for state, dist in self.__class__.transition_dists.items():
      self.transition_fns[state] = self.rng.dict_to_sampler(dist)

  def InitializeState(self):
    self.state = 0

  def LogProbability(self, state):
    return math.log(self.__class__.state_dist[state])

  def Propose(self):
    cls = self.__class__
    new_state = self.transition_fns[self.state]()
    logp_forward = math.log(cls.transition_dists[self.state][new_state])
    logp_backward = math.log(cls.transition_dists[new_state][self.state])
    return new_state, logp_forward, logp_backward


class CondTernaryMetropolisChain(TernaryMetropolisChain):
  """A 3-state Markov chain, conditioned on not visiting one state."""

  stationary_dist = {
      0: 0.2,
      1: 0.3
  }

  def LogProbability(self, state):
    if state in (0, 1):
      return super(CondTernaryMetropolisChain, self).LogProbability(state)
    else:
      return utils.LOG_PROB_0


class SlowTernaryMetropolisChain(TernaryMetropolisChain):
  """A version of the 3-state chain that converges more slowly."""

  proposal_dist = {
      0: .99,
      1: .005,
      2: .005
  }

  def __init__(self, rng=None):
    super(SlowTernaryMetropolisChain, self).__init__(rng=rng)
    self.proposal_fun = self.rng.dict_to_sampler(self.__class__.proposal_dist)

  def Propose(self):
    proposed_state = self.proposal_fun()
    logp_forward = math.log(self.proposal_dist[proposed_state])
    logp_backward = math.log(self.proposal_dist[self.state])
    return proposed_state, logp_forward, logp_backward


def LocalIsingLogProbability(state, temp, i, j, spin):
  """Returns the log-probability of a spin given its Markov blanket.

  This is intended for efficient computation of the ratio of
  probabilities between two Ising states. Given a fixed state, we
  compute the local state probability for spin=1 and spin=-1. This
  ratio is equal to the ratio of the entire states with state[i][j]
  set to -1/1.

  Args:
    state: a square matrix, with entries in {-1, 1}
    temp: temperature, a positive real number
    i: x-coordinate (int)
    j: y-coordinate (int)
    spin: -1 or 1

  Returns:
    log_prob: local log probability of state[i][j]==spin.
  """
  size = len(state)
  log_prob = 0.0
  if i > 0:
    log_prob += spin * state[i-1][j]
  if j > 0:
    log_prob += spin * state[i][j-1]
  if i < size - 1:
    log_prob += spin * state[i+1][j]
  if j < size - 1:
    log_prob += spin * state[i][j+1]
  return log_prob / temp


class IsingChain(mcmc.MetropolisHastingsChain):
  """A Markov chain that samples from a Ising model.

  A Ising model is a matrix of Boolean random variables, with variable
  values in {-1, 1}. Entries are connected to their (Manhatten)
  neighbors. Neighboring entries are probabilistically constrained to
  have the same value. This constraint is stronger at lower
  temperatures.

  Args:
    size: the width and height of the Ising matrix
    temp: temperature, a positive real number
  """

  def __init__(self, size, temp=1.0, rng=None):
    assert temp > 0.0
    self.size = size
    self.temp = temp
    super(IsingChain, self).__init__(rng=rng)

  def InitializeState(self):
    """Uniformly sample a size x size matrix with elements in {-1, 1}."""
    self.state = np.ones([self.size, self.size])
    for i in xrange(self.size):
      for j in xrange(self.size):
        if self.rng.flip(.5):
          self.state[i][j] = -1

  def InitializeStatistics(self):
    self.statistics = utils.SimpleNamespace(
        transitions=0, proposals=0, accepted=0)

  def Transition(self):
    """Execute Metropolis-Hastings proposals for all variables.

    Sweep over all entries with random row/column ordering. Propose
    flipping each entry. Accept according to MH rule.
    """
    self.statistics.transitions += 1
    for i in self.rng.random_permutation(self.size):
      for j in self.rng.random_permutation(self.size):
        self.statistics.proposals += 1
        local_state = self.state[i][j]
        cur_logp = LocalIsingLogProbability(
            self.state, self.temp, i, j, local_state)
        new_logp = LocalIsingLogProbability(
            self.state, self.temp, i, j, -local_state)
        log_acceptance_ratio = new_logp - cur_logp
        if log_acceptance_ratio >= 0 or self.rng.log_flip(log_acceptance_ratio):
          self.state[i][j] = -local_state
          self.statistics.accepted += 1

  def LogProbability(self, state):
    """Compute the total log probability of a Ising state."""
    log_prob = 0.0
    for i in xrange(self.size):
      for j in xrange(self.size):
        log_prob += LocalIsingLogProbability(
            state, self.temp, i, j, state[i][j])
    log_prob /= 2  # Sweeping over all variables double-counts each factor.
    return log_prob
