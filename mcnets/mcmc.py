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

"""Abstract interface for Markov chains.

This module provides abstract classes for general Markov chains and
for Metropolis-Hastings chains.
"""
import collections
import copy

from mcnets import utils


class MarkovChain(object):
  """Abstract class for Markov chains."""

  def __init__(self, rng=None):
    """Initialize Markov Chain state and random number generator.

    Args:
      rng: random number generator (an instance of utils.RandomState)
    """
    super(MarkovChain, self).__init__()
    # If no random state is given, set it nondeterministically
    # (expected behavior for random functions).
    self.rng = rng or utils.RandomState()
    self.InitializeState()
    self.InitializeStatistics()

  def InitializeState(self):
    """Assigns initial value to self.state."""
    raise NotImplementedError()

  def InitializeStatistics(self):
    self.statistics = utils.SimpleNamespace(transitions=0)

  def ResetState(self):
    self.InitializeState()

  def ResetStatistics(self):
    self.InitializeStatistics()

  def Reset(self):
    self.ResetState()
    self.ResetStatistics()

  def Transition(self):
    """Stochastically reassigns self.state."""
    raise NotImplementedError()

  def TransitionN(self, n):
    """Executes n state transitions."""
    for _ in xrange(n):
      self.Transition()


class MetropolisHastingsChain(MarkovChain):
  """Abstract class for Metropolis-Hastings chains."""

  def InitializeStatistics(self):
    """Resets acceptance rate statistics."""
    self.statistics = utils.SimpleNamespace(
        transitions=0, proposals=0, accepted=0)

  def LogProbability(self, state):
    """Scores a state, i.e., returns its prior log-probability.

    Args:
      state: a Markov chain state.

    Returns:
      logp: the log-probability of the given state.
    """
    raise NotImplementedError()

  def Propose(self):
    """Proposes a new state.

    Returns:
      proposed_state: a new state
      logp_forward: the log probability of returning proposed_state
        conditioned on self.state ("forward probability")
      logp_backward: the log probability of returning self.state conditioned
        on proposed_state ("backward probability").
    """
    raise NotImplementedError()

  def Transition(self):
    """Executes Metropolis-Hastings transition.

    Executes transition using LogProbability and Propose. Transitions
    to a new state with probability min(1, p(new_state)/p(old_state) *
    p_forward/p_backward). Updates acceptance rate statistics.
    """
    self.statistics.transitions += 1
    self.statistics.proposals += 1
    proposed_state, logp_forward, logp_backward = self.Propose()
    logp_new = self.LogProbability(proposed_state)
    logp_old = self.LogProbability(self.state)
    log_acceptance_ratio = logp_new - logp_old + logp_backward - logp_forward
    if log_acceptance_ratio >= 0 or self.rng.log_flip(log_acceptance_ratio):
      self.statistics.accepted += 1
      self.state = proposed_state


Revision = collections.namedtuple("Revision", ["params", "statistics"])


class HistoryMixin(object):
  """Extends a MarkovChain with a history of parameters and statistics.

  Each history entry is called a "revision". The Markov chain we
  extend is required to implement the methods GetParameters and
  SetParameters.

  Example of usage (order of superclasses doesn't matter):

  class MyMarkovChain(mcmc.MarkovChain, mcmc.HistoryMixin):

    def __init__(self, my_parameter):
      self.my_parameter = my_parameter
      super(MyMarkovChain, self).__init__()

    def GetParameters(self):
      return dict(my_parameter=my_parameter)

    def SetParameters(self, params):
      self.my_parameter = params["my_parameter"]

  >>> chain = MyMarkovChain(my_parameter=5)
  >>> chain.RecordCurrentRevision()
  >>> chain.my_parameter = 7
  >>> chain.RestoreRevision(chain.PreviousRevision())
  >>> chain.my_parameter
  5
  """

  def __init__(self, *args, **kwargs):
    super(HistoryMixin, self).__init__(*args, **kwargs)
    self.InitializeHistory()

  def InitializeHistory(self):
    self.history = []

  def ResetHistory(self):
    self.InitializeHistory()

  def RecordCurrentRevision(self):
    self.history.append(self.CurrentRevision())

  def CurrentRevision(self):
    revision = Revision(
        params=self.GetParameters(),
        statistics=copy.deepcopy(self.statistics))
    return revision

  def GetRevision(self, revision_index):
    assert len(self.history) > revision_index
    return self.history[revision_index]

  def HasPreviousRevision(self):
    return bool(self.history)

  def PreviousRevision(self):
    assert self.HasPreviousRevision()
    return self.history[-1]

  def RestoreRevision(self, revision):
    self.SetParameters(revision.params)
    self.statistics = revision.statistics
