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

from mcnets import models
from mcnets import testutils


class TestIsingChain(testutils.StochasticTestCase):

  def testInitialization(self):
    """Test that approximately half of the entries are -1, half are 1."""
    def SampleInitialStateSummary():
      chain = models.IsingChain(size=100, temp=1.0, rng=self.rng)
      fraction_spin_up = sum(sum(chain.state + 1)) / (2 * chain.size**2)
      return fraction_spin_up

    self.assertHasMean(
        thunk=SampleInitialStateSummary,
        true_mean=0.5,
        nsamples=1000,
        margin=0.03)

  def testTransitions(self):
    """Test that transitions stochastically increase the state probability."""
    def SampleTransitionSummary():
      chain = models.IsingChain(size=20, temp=1.0, rng=self.rng)
      pre_logp = chain.LogProbability(chain.state)
      chain.TransitionN(20)
      post_logp = chain.LogProbability(chain.state)
      return 1 if post_logp > pre_logp else 0

    n_logp_increased = sum(SampleTransitionSummary() for _ in range(50))
    self.assertGreater(n_logp_increased, 40)

  def testLogProbability(self):
    """Test that fast local scoring coincides with accurate global scoring."""
    size = 5
    for i in range(size):
      for j in range(size):
        chain = models.IsingChain(size=size, temp=1.0, rng=self.rng)
        local_pre_logp = models.LocalIsingLogProbability(
            state=chain.state,
            temp=chain.temp,
            i=i, j=j,
            spin=chain.state[i][j])
        local_post_logp = models.LocalIsingLogProbability(
            state=chain.state,
            temp=chain.temp,
            i=i, j=j,
            spin=-chain.state[i][j])
        local_ratio = local_post_logp - local_pre_logp
        global_pre_logp = chain.LogProbability(chain.state)
        chain.state[i][j] = -chain.state[i][j]
        global_post_logp = chain.LogProbability(chain.state)
        global_ratio = global_post_logp - global_pre_logp
        self.assertAlmostEqual(local_ratio, global_ratio)


if __name__ == "__main__":
  unittest.main()
