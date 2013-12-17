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

from mcnets import models
from mcnets import testutils


class TestMarkovChain(testutils.StochasticTestCase):

  def GetSimulator(self, chain_class):
    """Returns a thunk that runs the Markov chain, returning its state."""
    def Simulator():
      chain = chain_class(rng=self.rng)
      chain.TransitionN(self.ntransitions)
      return chain.state
    return Simulator

  def setUp(self):
    self.nrepeats = 1000
    self.ntransitions = 1000
    self.tolerance = 0.05

  def testConvergence(self):
    self.assertHasDistribution(
        self.GetSimulator(models.BinaryMarkovChain),
        models.BinaryMarkovChain.stationary_dist,
        self.nrepeats,
        self.tolerance)


class TestMetropolisHastingsChain(testutils.StochasticTestCase):

  def GetSimulator(self, chain_class):
    """Returns a thunk that runs the Markov chain, returning its state."""
    def Simulator():
      chain = chain_class(rng=self.rng)
      chain.TransitionN(self.ntransitions)
      self.assertEqual(chain.statistics.transitions, self.ntransitions)
      self.assertGreater(chain.statistics.accepted, 0)
      self.assertLess(chain.statistics.accepted, self.ntransitions)
      self.assertEqual(chain.statistics.proposals, chain.statistics.transitions)
      return chain.state
    return Simulator

  def setUp(self):
    self.nrepeats = 1000
    self.ntransitions = 1000
    self.tolerance = 0.05

  def testUnconditionalConvergence(self):
    self.assertHasDistribution(
        self.GetSimulator(models.TernaryMetropolisChain),
        models.TernaryMetropolisChain.stationary_dist,
        self.nrepeats,
        self.tolerance)

  def testConditionalConvergence(self):
    self.assertHasDistribution(
        self.GetSimulator(models.CondTernaryMetropolisChain),
        models.CondTernaryMetropolisChain.stationary_dist,
        self.nrepeats,
        self.tolerance)


class TestHistoryMixin(testutils.StochasticTestCase):

  def setUp(self):
    self.initial_p_flip = .3
    self.chain = models.ParameterizedBinaryMarkovChain(
        p_flip=self.initial_p_flip)
    self.ntransitions = 1000

  def testHistory(self):
    self.assertFalse(self.chain.HasPreviousRevision())
    # Recording a revision doesn't change its state.
    revision_1 = self.chain.CurrentRevision()
    self.chain.RecordCurrentRevision()
    self.assertTrue(self.chain.HasPreviousRevision())
    revision_2 = self.chain.PreviousRevision()
    self.assertEqual(revision_1.params.p_flip, revision_2.params.p_flip)
    self.assertEqual(revision_1.statistics.transitions,
                     revision_2.statistics.transitions)
    # Running a chain changes the state of statistics, but doesn't
    # change other revision parameters.
    self.chain.TransitionN(self.ntransitions)
    revision_3 = self.chain.CurrentRevision()
    self.assertEqual(revision_2.params.p_flip, revision_3.params.p_flip)
    self.assertNotEqual(revision_2.statistics.transitions,
                        revision_3.statistics.transitions)
    self.chain.RecordCurrentRevision()
    # Changing parameters is correctly reflected in revisions.
    self.chain.p_flip = .5
    self.chain.TransitionN(self.ntransitions)
    self.chain.RecordCurrentRevision()
    revision_4 = self.chain.PreviousRevision()
    self.assertNotEqual(revision_3.params.p_flip, revision_4.params.p_flip)
    self.assertNotEqual(revision_3.statistics.transitions,
                        revision_4.statistics.transitions)
    # Restoring parameters works.
    self.chain.RestoreRevision(revision_3)
    revision_5 = self.chain.CurrentRevision()
    self.assertEqual(revision_5.params.p_flip, self.initial_p_flip)
    self.assertEqual(revision_5.statistics.transitions,
                     revision_3.statistics.transitions)


if __name__ == "__main__":
  unittest.main()
