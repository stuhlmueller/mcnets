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

"""Iterative adaptation of Markov chain networks.

Strategy: Start with a densely connected, uniformly weighted graph of
swap frequencies, then adapt both chain parameters and connection
weights based on which chains and connections contribute most to
overall flow, i.e. to replica roundtrips from parameters (0, 0, ...)
to (1, 1, ...) and back.

Repeat the following steps:

(1) Collect flow data on current configuration

Run the network chain, i.e., local transitions + swaps, until a
given number of replica roundtrips has been reached. As a global
score, record the time this took. Locally, for each chain and graph
edge, keep a score that reflects how often this element was present in
successful roundtrips. Whenever a replica completes a roundtrip,
increment the score of each chain that touched this replica and each
edge that was successfully traversed by 1/(length of roundtrip). The
length of a roundtrip is the total number of chains that touched the
replica.

(2) Collect baseline flow data

We cannot directly use the scores collected in (1) to improve our
configuration. To see this, consider a chain that resides in a part of
the swap graph that has weak connections to the rest of the graph. We
are unlikely to propose to swap with this chain. As a consequence, the
number of replica roundtrips that touch this chain will be lower than
for more well-connected regions of the graph. This fact is not
evidence per se that this chain is less useful, and that we should
propose swaps with it even less frequently. To evaluate whether we
should increase or decrease connectivity to this chain, we need to
look at whether this chain did better or worse than what we would
expect given our swap weights. For this purpose, we simulate flow in
the weighted graph without running any base transitions. In this
simulation, we accept all swap proposals. This results in baseline
scores to compare to.

(3) Check for harmful changes, revert if necessary

Check whether the global score is as good or better than it was for
the previous generation. If it deteriorated, revert to the previous
generation's parameters and continue with (1) to refine the previous
score estimate.

(4) Update configuration

We have to trade off the desire for fast improvement -- suggesting big
changes -- against the fact that different changes interact --
suggesting small incremental changes. As a preliminary response to
this tradeoff, we update either connection weights or the component
chain parameters in each generation, but not both simultaneously:

(4a) Adapt connection weights (swap frequencies, a nxn matrix)

Up/downweight connections based on whether they have been more/less
useful than expected.

(4b) Adapt chain parameters (temperatures)

Remove the chain with lowest relative score. Make a randomly varied
copy of the most useful chain (relative what we would expect ot see),
including a copy of its connections.

TODO(stuhlmueller): Make updates more principled.
TODO(stuhlmueller): Initialize scores to epsilon (maybe 1 / num_edges).
"""
from __future__ import division

import collections
import copy

import numpy as np

from mcnets import mcmc
from mcnets import utils


class NetworkChain(mcmc.MarkovChain):
  """Executes swaps based on a graph of swap weights."""

  def __init__(self, chains, nsteps_per_sweep, nswaps_per_sweep,
               burn_roundtrips=0, swap_graph=None, rng=None):
    """Initializes network of chains given component chains.

    Args:
      chains: list of component chains, including chains with extremal
        params. Chains are required to have a "temps" attribute.
      nsteps_per_sweep: the number of transitions to apply to each
        component chain each time the network chain transition method
        is called.
      nswaps_per_sweep: the number of swaps to execute each time the
        network chain transition method is called.
      burn_roundtrips: the number of replica roundtrips required
        before we start updating statistics about per-chain and
        per-connection roundtrip counts.
      swap_graph: a nxn symmetric 2D numpy array, with n equal
        to the number of chains
      rng: a random stream (utils.RandomState)
    """
    assert len(chains) > 1, len(chains)
    self.CheckTemperatures(chains)
    self.chains = chains
    self.nsteps_per_sweep = nsteps_per_sweep
    self.nswaps_per_sweep = nswaps_per_sweep
    self.burn_roundtrips = burn_roundtrips
    super(NetworkChain, self).__init__(rng=rng)
    if swap_graph is None:  # Can't implicitly coerce numpy array to Boolean.
      self.swap_graph = self.GetInitialSwapGraph()
    else:
      self.swap_graph = swap_graph

  def GetInitialSwapGraph(self):
    """Returns fully connected graph of swap connections."""
    swap_graph = utils.SymmetricOnes(len(self.chains))
    swap_graph -= np.diag(swap_graph.diagonal())  # No self-swaps
    return swap_graph

  def InitializeState(self):
    """The chain doesn't keep state beyond the state of its component chains."""
    for chain in self.chains:
      chain.InitializeState()

  def CheckTemperatures(self, chains):
    """Checks global consistency of chain params: extremal chains, ordered."""
    assert len(chains) > 1, len(chains)
    for chain in chains:
      assert hasattr(chain, "temps")
      for temp in chain.temps:
        assert 0 <= temp <= 1, temp
    for temp in chains[0].temps:
      assert temp == 0.0
    for temp in chains[-1].temps:
      assert temp == 1.0

  def InitializeStatistics(self):
    """Initializes acceptance, replica behavior, and score statistics.

    Statistics:
      transitions: the total number of transitions; each transition
        consists of len(self.chains)*self.nsteps_per_sweep base moves
        and self.nswaps_per_sweep swap moves.
      replica: a record with statistics that describe the behavior of
        replica, i.e., the movement of states across different
        temperatures (see InitializeReplicaStatistics).
      chain_scores: an array with n entries, one for each chain.
      edge_scores: a symmetric nxn matrix, one entry for each edge.
    """
    n = len(self.chains)
    self.statistics = utils.SimpleNamespace(
        transitions=0,
        chain_scores=np.zeros(n),
        edge_scores=utils.SymmetricZeros(n),
        replica=None)
    self.InitializeReplicaStatistics()
    for chain in self.chains:
      chain.InitializeStatistics()

  def InitializeReplicaStatistics(self):
    """Initializes replica behavior statistics.

    For a densely connected swap graph of n chains, storing replica
    traversal statistics requires O(n^3) space. This can be
    improved-upon by replacing the initial swap graph with a sparse
    graph.

    Statistics:
      ids: maps chain index to id of replica currently located at
        chain
      directions: maps replica id to {-1, 0, 1}, depending on whether
        replica last visited bottom-most chain (1, "upwards"), top
        chain (-1, "downwards"), or neither (0).
      traversed: maps replica id to a map from (undirected)
        edges--i.e., pairs of chain indices--to counts that indicates
        how often this edge has been traversed by the replica on the
        current roundtrip.
      roundtrips: the total number of replica that made it all the way
        from the top to the bottom
    """
    n = len(self.chains)
    replica_stats = utils.SimpleNamespace(
        ids=range(n),
        directions=np.zeros(n),
        traversed=[collections.defaultdict(lambda: 0) for _ in xrange(n)],
        roundtrips=0)
    self.statistics.replica = replica_stats

  def ResetReplicaStatistics(self):
    """Resets replica statistics to initial state."""
    self.InitializeReplicaStatistics()

  def UpdateReplicaStatistics(self, swap_accepted, chain1_index, chain2_index):
    """Updates information about replica behavior after a swap.

    Args:
      swap_accepted: a Boolean indicating whether the swap was accepted
      chain1_index: id of first chain participating in swap (integer)
      chain2_index: id of second chain participating in swap (integer)
    """
    if not swap_accepted:
      return
    r = self.statistics.replica
    # Swap the replica ids.
    r.ids[chain1_index], r.ids[chain2_index] = (
        r.ids[chain2_index], r.ids[chain1_index])
    # Update edge traversal counts for both replicas.
    key = tuple(sorted([chain1_index, chain2_index]))
    r.traversed[r.ids[chain1_index]][key] += 1
    r.traversed[r.ids[chain2_index]][key] += 1
    # Check if we have completed a roundtrip.
    if r.directions[r.ids[0]] == -1:
      # Update number of roundtrips
      r.roundtrips += 1
      if r.roundtrips > self.burn_roundtrips:
        # Increment score of all chains and edges that were touched
        # during this roundtrip by 1/(length of roundtrip).
        traversal_counts = r.traversed[r.ids[0]]
        roundtrip_length = sum(traversal_counts.values())
        traversed_chain_indices = set([i for (i, _) in traversal_counts] +
                                      [j for (_, j) in traversal_counts])
        for chain_index in traversed_chain_indices:
          self.statistics.chain_scores[chain_index] += 1/roundtrip_length
        for edge_index in traversal_counts:
          self.statistics.edge_scores[edge_index] += 1/roundtrip_length
      # For the replica that completed the roundtrip, reset traversal
      # counts for all edges to 0.
      r.traversed[r.ids[0]] = collections.defaultdict(lambda: 0)
    # Update directions of replicas at top/bottom chains.
    r.directions[r.ids[0]] = 1
    r.directions[r.ids[-1]] = -1

  def SwapTransition(self, chain1_index, chain2_index):
    """Proposes to swap the states of two chains, accepts based on MH rule."""
    assert chain1_index != chain2_index
    chain1, chain2 = self.chains[chain1_index], self.chains[chain2_index]
    logp_old = (chain1.LogProbability(chain1.state) +
                chain2.LogProbability(chain2.state))
    logp_new = (chain1.LogProbability(chain2.state) +
                chain2.LogProbability(chain1.state))
    log_acceptance_ratio = logp_new - logp_old
    swap_accepted = (log_acceptance_ratio >= utils.LOG_PROB_1 or
                     self.rng.log_flip(log_acceptance_ratio))
    if swap_accepted:
      chain1.state, chain2.state = chain2.state, chain1.state
    self.UpdateReplicaStatistics(swap_accepted, chain1_index, chain2_index)
    return swap_accepted

  def BaseTransition(self):
    """Applies transition operator to component chains."""
    for chain in self.chains:
      chain.TransitionN(self.nsteps_per_sweep)

  def GetSwapIndexSampler(self):
    """Constructs a function that samples pairs of chain indices."""
    n = len(self.chains)
    nconnections = (n * (n - 1)) / 2
    weights = np.zeros(shape=nconnections)
    indices = np.zeros(shape=nconnections, dtype=(int, 2))
    k = 0
    for i in xrange(n):
      for j in xrange(i + 1, n):
        indices[k] = (i, j)
        weights[k] = self.swap_graph[(i, j)]
        k += 1
    assert k == nconnections
    return self.rng.list_to_sampler(indices, weights)

  def Transition(self):
    """Executes nsteps_per_sweep transitions for each chain, then swaps."""
    for chain in self.chains:
      chain.TransitionN(self.nsteps_per_sweep)
    sample_swap_indices = self.GetSwapIndexSampler()
    for _ in xrange(self.nswaps_per_sweep):
      i, j = sample_swap_indices()
      self.SwapTransition(i, j)
    self.statistics.transitions += 1

  def TransitionForRoundtrips(self, nroundtrips):
    """Execute transitions until a given number of roundtrips is reached."""
    n = self.statistics.replica.roundtrips
    while self.statistics.replica.roundtrips - n < nroundtrips:
      self.Transition()

  @property
  def state(self):
    """Only the state of the base chain is visible externally."""
    return self.chains[0].state


class AdaptiveNetworkChain(NetworkChain, mcmc.HistoryMixin):
  """Iteratively adapts a swap configuration for parallel tempering."""

  param_names = [
      "nsteps_per_sweep",
      "nswaps_per_sweep",
      "burn_roundtrips",
      "swap_graph",
      "temps"]

  Params = collections.namedtuple(
      "Params", param_names)

  def NoisifyTemperatures(self, temps, pseudocounts=200):
    """Returns a stochastic variation of the given list of temperatures.

    For a temperature with value p, this samples new parameters from a
    Beta(1+p*pseudocounts, 1+(1-p)*pseudocounts) distribution. This
    ensures that new parameters are close to old ones, that all
    parameters are restricted to [0, 1], and that the corner case
    where p=0 or p=1 does not cause an error. A pseudocount of 200
    corresponds to a standard deviation around .035.

    Args:
      temps: list of temperatures (positive real numbers)
      pseudocounts: Beta parameter that affects variance of new
        temperatures.

    Returns:
      new_temps: a noisy copy of temps
    """
    new_temps = [self.rng.beta(t*pseudocounts+1, (1-t)*pseudocounts+1)
                 for t in temps]
    return new_temps

  def AdaptTemperatures(self):
    """Replaces least useful temps with noisy copy of most useful temps.

    Replace parameters of least useful chain with noisified version of
    parameters of most useful chain. Copy connections from most useful
    chain to least useful chain.
    """
    min_index = np.argmin(self.statistics.chain_scores)
    max_index = np.argmax(self.statistics.chain_scores)
    min_chain = self.chains[min_index]
    max_chain = self.chains[max_index]
    new_temps = self.NoisifyTemperatures(max_chain.temps)
    print (("Replacing temps of chain %i (%s) with noisified "
            "temps of chain %i (%s), copying connections.") %
           (min_index, min_chain.temps, max_index, new_temps))
    min_chain.temps = new_temps
    for i in range(len(self.chains)):
      if i == min_index:
        # No self swaps.
        self.swap_graph[min_index, min_index] = 0
      elif i == max_index:
        # The most useful chain doesn't have a self-edge, but we want
        # an edge from the chain being modified to this chain, so we
        # make up an edge weight.
        self.swap_graph[min_index, max_index] = 1
      else:
        # Copy edge weights from most useful chain.
        self.swap_graph[min_index, i] = self.swap_graph[max_index, i]

  def AdaptSwapGraph(self, baseline_statistics):
    """Adapts weights based on actual vs expected contribution to roundtrips.

    The purpose of this update rule is to adjust the connectivity
    pattern to direct more proposals to chains that are more
    frequently part of successful roundtrips than expected from the
    connectivity pattern alone. The interaction with the temperature
    adaptation rule results in a vaguely EM-like algorithm: we
    optimize connectivity for the current temperatures/chains, then
    replace the least useful chains, optimize connectivity again, etc.

    Args:
      baseline_statistics: statistics for a NetworkChain that serve as
        comparison for adaptation relative to expectations.
    """
    for (i, j), empirical_score in np.ndenumerate(self.statistics.edge_scores):
      if i > j:  # Only need to modify entries in one triangle due to symmetry.
        baseline_score = baseline_statistics.edge_scores[i, j]
        if baseline_score == 0.0 and empirical_score == 0.0:
          # For edges that are not expected to be useful, and are not
          # useful in practice, no modification is necessary.
          continue
        elif baseline_score == 0.0:
          # If an edge is not expected to be useful at all, but turns
          # out to be useful, we want to increase its usage. But by
          # how much? The ratio 2.0 here is arbitrary.
          ratio = 2.0
        else:
          ratio = empirical_score / baseline_score
        print ("Multiplying edge %s (weight %s) by ratio %s (%f / %f)." %
               ((i, j), self.swap_graph[i, j], ratio, empirical_score,
                baseline_score))
        self.swap_graph[i, j] *= ratio

  def Adapt(self, generations, roundtrips_per_generation, adapt_temps_every=2):
    """Repeatedly adapts temperatures and swap freqs to optimize replica flow.

    We simulate the swap dynamics of the network chain in a setting
    where all swaps are accepted. This results in a baseline score for
    each edge. We normalize and compare to empirical scores. For some
    edges, the actual score is going to be lower (indicating that this
    edge was used in fewer roundtrips than expected), whereas for
    others, it is going to be higher (edge used in more roundtrips
    than expected). We upweight edges that are more useful than
    expected, and downweight edges that are less useful. We replace
    chains that are not useful (on an absolute scale) with copies of
    chains that are most useful.

    Args:
      generations: number of generations (int).
      roundtrips_per_generation: number of roundtrips for data collection (int).
      adapt_temps_every: update temps every nth generation.
    """
    # Import stateless here to prevent circular dependence.
    from mcnets import stateless
    # Create a version of this chain that accepts all swap proposals:
    stateless_chain = stateless.StatelessNetworkChain(
        swap_graph=self.swap_graph,
        burn_roundtrips=self.burn_roundtrips)
    adapted_in_previous_generation = False
    for i in xrange(generations):
      # Collect replica flow data on current setup.
      self.TransitionForRoundtrips(roundtrips_per_generation)
      # Collect baseline flow data
      stateless_chain.TransitionForRoundtrips(roundtrips_per_generation)
      # Check for harmful changes, revert if necessary.
      roundtrip_cost = self.TransitionsPerRoundtrip()
      print ("Generation %i/%i (%f transitions per roundtrip, %i total)" %
             (i+1, generations, roundtrip_cost, self.statistics.transitions))
      print "Graph:\n", self.swap_graph
      print "Temps: ", [chain.temps for chain in self.chains]
      if adapted_in_previous_generation:
        previous_roundtrip_cost = self.TransitionsPerRoundtrip(
            self.PreviousRevision().statistics)
        if roundtrip_cost > previous_roundtrip_cost:
          print "Cost increased, reverting to previous settings."
          self.RestoreRevision(self.PreviousRevision())
          adapted_in_previous_generation = False
          continue
      # Store checkpoint for current parameters and statistics before
      # we make any changes.
      self.RecordCurrentRevision()
      # Adapt swap frequencies or chain parameters.
      if i % adapt_temps_every == 0:
        self.AdaptTemperatures()
      else:
        self.AdaptSwapGraph(stateless_chain.statistics)
        stateless_chain.ResetStatistics()
        stateless_chain.swap_graph = self.swap_graph
      adapted_in_previous_generation = True
      self.ResetStatistics()

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

  def GetParameters(self):
    """Returns a record that captures the current chain parameterization."""
    params = self.__class__.Params(
        nsteps_per_sweep=self.nsteps_per_sweep,
        nswaps_per_sweep=self.nswaps_per_sweep,
        burn_roundtrips=self.burn_roundtrips,
        swap_graph=copy.copy(self.swap_graph),
        temps=[chain.temp for chain in self.chains]
    )
    return params

  def SetParameters(self, params):
    """Sets chain parameters based on record with parameter values.

    Args:
      params: a record with parameter names and values, usually
        created by GetParameters.
    """
    self.nsteps_per_sweep = params.nsteps_per_sweep
    self.nswaps_per_sweep = params.nswaps_per_sweep
    self.burn_roundtrips = params.burn_roundtrips
    self.swap_graph = copy.copy(params.swap_graph)
    for chain, temp in zip(self.chains, params.temps):
      chain.temp = temp
