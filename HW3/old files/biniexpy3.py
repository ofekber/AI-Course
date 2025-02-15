from collections import deque
from math import inf

# IDs as requested
ids = [ "208912675", "318159506" ]

# Rewards / penalties
DESTROY_HORCRUX_REWARD = 2
RESET_REWARD = -2
DEATH_EATER_CATCH_REWARD = -1


def is_passable(cell_char):
    """Return True if the cell is passable (not an impassable 'i' or 'I')."""
    return cell_char.lower() != 'i'


def get_neighbors(r, c, grid):
    """Return all passable neighbors (up/down/left/right) of (r,c)."""
    rows = len(grid)
    cols = len(grid[ 0 ]) if rows > 0 else 0
    results = [ ]
    for (dr, dc) in [ (-1, 0), (1, 0), (0, -1), (0, 1) ]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            if is_passable(grid[ rr ][ cc ]):
                results.append((rr, cc))
    return results


###############################################################################
# Non-negative BFSWizard
###############################################################################
class BFSWizard:
    """
    Offline BFS that ensures any partial path that dips below 0 expected score is pruned.
    We store a policy so that we can return an action in `act()`.

    State representation for BFS:
      (turns_left, wizardPositions, destroyedHorcruxSet, scoreSoFar)
    We expand possible actions, compute immediate reward (destroy, reset, or collisions),
    and skip expansions where newScore < 0. The BFS stops at t=0 or when no expansions remain.

    Collisions use a simple 1-step look-ahead probability model,
    and each wizard collision yields DEATH_EATER_CATCH_REWARD = -1 to the running total.
    """

    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.grid = initial_state[ "map" ]
        self.rows = len(self.grid)
        self.cols = len(self.grid[ 0 ]) if self.rows > 0 else 0

        # Basic references
        self.wizard_names = list(initial_state[ "wizards" ].keys())
        self.horcrux_names = list(initial_state[ "horcrux" ].keys())

        # Store Death Eater paths
        self.death_eater_paths = {
            dname: ddata[ "path" ]
            for dname, ddata in initial_state[ "death_eaters" ].items()
        }

        # Number of total turns we have
        self.turn_limit = initial_state[ "turns_to_go" ]

        # We'll build an offline BFS plan in the constructor
        # policy:  (t, wpos, destroyedSet, scoreSoFar) -> bestAction
        # visited: (t, wpos, destroyedSet) -> bestScoreSoFar
        #  (We store the best possible partial score for that sub-state, ignoring "scoreSoFar" dimension.)
        self.policy = {}
        self.visited = {}

        self._build_and_solve_bfs()

    def _build_and_solve_bfs(self):
        """
        Perform BFS from the start node (turns_to_go, initialWizardPositions, emptyDestroyedSet, 0 score).
        Keep only expansions that remain >= 0 in scoreSoFar.
        Among expansions for a given (t, wpos, destroyedSet), we store whichever has the best partial score.
        That means we do not expand from a node if we've found a strictly better partial score for it before.
        """

        # The initial wizard positions:
        init_wpos = tuple(
            wdata[ "location" ] for wdata in self.initial_state[ "wizards" ].values()
        )
        # No Horcrux destroyed yet
        start_destroyed = frozenset()
        start_score = 0.0

        start_node = (self.turn_limit, init_wpos, start_destroyed, start_score)

        # We say visited[(t, wpos, destroyedSet)] = bestSoFar
        self.visited[ (self.turn_limit, init_wpos, start_destroyed) ] = start_score

        queue = deque([ start_node ])

        while queue:
            (t, wpos, destroyedSet, scoreSoFar) = queue.popleft()

            if t <= 0:
                # No further expansions
                continue

            # Enumerate all feasible *global* actions
            possible_actions = self._enumerate_global_actions(wpos, destroyedSet)

            for action in possible_actions:
                # Apply the action to get next state
                next_t = t - 1
                next_wpos, next_destroyedSet, immediate_reward = self._apply_action(
                    wpos, destroyedSet, action
                )
                # Also add collision penalty (expected)
                collision_penalty = self._expected_collision_penalty(next_wpos)
                new_score = scoreSoFar + immediate_reward + collision_penalty

                # Prune expansions that go negative
                if new_score < 0:
                    continue

                # Check if we found a strictly better partial score for (next_t, next_wpos, next_destroyedSet)
                next_key = (next_t, next_wpos, next_destroyedSet)
                old_best = self.visited.get(next_key, -inf)
                if new_score > old_best:
                    self.visited[ next_key ] = new_score
                    # Record policy so we can reconstruct the best action for that sub-state
                    # We store the policy keyed by the FULL state (including new_score),
                    # or we could store a simpler key. For simplicity, we do:
                    self.policy[ (next_t, next_wpos, next_destroyedSet, new_score) ] = \
                        ((t, wpos, destroyedSet, scoreSoFar), action)

                    # Enqueue the newly discovered (nonnegative) partial plan
                    queue.append((next_t, next_wpos, next_destroyedSet, new_score))

    def _expected_collision_penalty(self, wizPositions):
        """
        A 1-step approximation: For each wizard, compute the probability that a Death Eater
        will occupy the same tile next turn. Then the expected penalty is sum of -1 * p for each wizard.
        """
        total_penalty = 0.0
        for wpos in wizPositions:
            p_collision = self._death_eater_collision_prob(wpos)
            # Each collision yields -1, so expected penalty = -1 * p_collision
            total_penalty += p_collision * DEATH_EATER_CATCH_REWARD
        return total_penalty

    def _death_eater_collision_prob(self, cell):
        """
        One-step collision probability for a single tile.
        For each Death Eater, it can move to index-1, index, or index+1 with equal probability
        (unless at the ends). If any DE has path[node] == cell, we consider that event’s probability.
        Then p(not occupied) = product(1 - p(DE occupies cell)), etc.
        Here, we do:  p(occupied by >= 1 DE) = 1 - product(1 - p(DE_i on cell))
        """
        death_eaters = self.initial_state[ "death_eaters" ]
        p_not_occupied = 1.0

        for dname, ddata in death_eaters.items():
            path = self.death_eater_paths[ dname ]
            idx = ddata[ "index" ]
            # possible next indices
            candidates = [ idx ]
            if idx > 0:
                candidates.append(idx - 1)
            if idx < len(path) - 1:
                candidates.append(idx + 1)

            p_this_de = 0.0
            each_prob = 1.0 / len(candidates)
            for cidx in candidates:
                if path[ cidx ] == cell:
                    p_this_de += each_prob

            p_not_occupied *= (1.0 - p_this_de)

        return 1.0 - p_not_occupied

    def _enumerate_global_actions(self, wizPositions, destroyedSet):
        """
        Return a list of possible *global* actions:
          1) "terminate"
          2) "reset"
          3) A "joint action": a tuple of sub-actions, one per wizard.
             Sub-action examples: ("move", wName, (r,c)), ("destroy", wName, hName), ("wait", wName)
        We'll do a simple approach: each wizard can "move" to a neighbor, "destroy" (if on a horcrux),
        or "wait".
        """
        import itertools

        actions = [ "terminate", "reset" ]

        # Build per-wizard sub-action sets
        wSubActions = [ ]
        for w_idx, pos in enumerate(wizPositions):
            wname = self.wizard_names[ w_idx ]
            r, c = pos
            subA = [ ]

            # Move to neighbors
            for nbr in get_neighbors(r, c, self.grid):
                subA.append(("move", wname, nbr))

            # Destroy if there's *any* horcrux at this location (that isn't in destroyedSet).
            # In your environment, you might track which specific horcrux is at (r,c).
            # For demonstration, we just add "destroy" if there's a horcrux-loc matching this pos
            # (not in destroyedSet).
            for hname in self.horcrux_names:
                if hname not in destroyedSet:
                    # If that horcrux is still alive, check if it's at (r,c)
                    hx_loc = self.initial_state[ "horcrux" ][ hname ][ "location" ]
                    # or if it can teleport, you'd do more advanced checking.
                    # For simplicity, we'll just see if "current location" == wloc
                    if hx_loc == pos:
                        subA.append(("destroy", wname, hname))

            # Wait
            subA.append(("wait", wname))
            wSubActions.append(subA)

        # Now cartesian product
        for combo in itertools.product(*wSubActions):
            actions.append(combo)

        return actions

    def _apply_action(self, wizPositions, destroyedSet, action):
        """
        Apply the given *global* action to produce (next_wizPositions, next_destroyedSet, immediateReward).
        Collision penalty is handled separately in _expected_collision_penalty().
        """
        # 1) If action == "terminate":
        if action == "terminate":
            # No position change, no new destroyed set, no immediate reward
            return (wizPositions, destroyedSet, 0.0)

        # 2) If action == "reset":
        if action == "reset":
            initWpos = tuple(wd[ "location" ] for wd in self.initial_state[ "wizards" ].values())
            # Usually reset does not "un-destroy" Horcruxes, unless your environment says otherwise.
            # We'll assume destroyedSet stays the same.
            return (initWpos, destroyedSet, RESET_REWARD)

        # 3) Otherwise, it's a tuple of sub-actions
        if isinstance(action, tuple) and len(action) > 0 and isinstance(action[ 0 ], str):
            # single sub-action => treat it as list of one
            joint_actions = [ action ]
        else:
            joint_actions = list(action)

        newPos = list(wizPositions)
        newDestroyedSet = set(destroyedSet)
        total_rew = 0.0

        for sub in joint_actions:
            atype = sub[ 0 ]
            if atype == "move":
                # ("move", wName, (nr,nc))
                _, wName, (nr, nc) = sub
                w_idx = self.wizard_names.index(wName)
                newPos[ w_idx ] = (nr, nc)

            elif atype == "destroy":
                # ("destroy", wName, hName)
                _, wName, hName = sub
                # only reward if that horcrux isn't already destroyed
                if hName not in newDestroyedSet:
                    newDestroyedSet.add(hName)
                    total_rew += DESTROY_HORCRUX_REWARD

            elif atype == "wait":
                pass

        return (tuple(newPos), frozenset(newDestroyedSet), total_rew)

    def act(self, state):
        """
        Use the BFS policy to decide the next action. Because our BFS stored states as
         (t, wpos, destroyedSet, scoreSoFar),
        we need to find which 'scoreSoFar' we are at. The environment might not store that.
        So we can't do an *exact* lookup unless we track the running sum in the environment.

        One workaround:
         - We can look up the 'best known' partial score in visited for (t, wpos, destroyedSet),
           then find the policy entry that matches that best partial score.
         - If we can't find it, we do a fallback (like "terminate" or a simpler BFS step).
        """
        t = state[ "turns_to_go" ]
        if t <= 0:
            return "terminate"

        # Wizard positions (tuple)
        wpos = tuple(st[ "location" ] for st in state[ "wizards" ].values())

        # destroyedSet: we only know if the environment is tracking which horcruxes are destroyed.
        # Some environments track that, others do not. We'll assume `state["horcrux"]` has only
        # the *remaining* ones. Let's build a set of destroyed horcrux names by seeing which
        # ones are missing from the initial list.
        # In your environment, you might have a direct list of destroyed ones.
        existing_hnames = set(state[ "horcrux" ].keys())
        all_hnames = set(self.horcrux_names)
        destroyed = all_hnames - existing_hnames
        destroyed_fs = frozenset(destroyed)

        # Now let's see if we visited (t, wpos, destroyed_fs).
        # If not, we fallback
        key = (t, wpos, destroyed_fs)
        if key not in self.visited:
            # fallback
            return "terminate"  # or do some local logic

        best_score = self.visited[ key ]
        # We want the policy that stored (t, wpos, destroyed_fs, best_score)
        policy_key = (t, wpos, destroyed_fs, best_score)
        if policy_key not in self.policy:
            # fallback
            return "terminate"

        # Otherwise, fetch the recommended action
        # self.policy[(next_t, next_wpos, next_destroyedSet, new_score)] =
        #   ((t, wpos, destroyedSet, scoreSoFar), action)
        # We only stored a "back pointer" from the child.
        # But let's invert it: we want the action from the *parent*.
        # So let's do a quick search. Alternatively, we could store forward pointers.

        # Actually from the BFS building code:
        #   self.policy[(child_t, child_wpos, child_destroyedSet, child_score)] = ((t, wpos, dSet, scSoFar), action)
        # means the "child" is the key in the dictionary. The stored value is (parentState, action).
        # So if we want the action for the *current* state, we need to search among children.
        # That can be tricky. Instead, we can do the BFS differently: store
        # policy[(t, wpos, destroyedSet, scoreSoFar)] = bestAction

        # Let's do it the simpler way: we can modify the BFS code so that *parent* state is the key.
        # For demonstration, let's do a quick fix search:

        # We'll search the dictionary for an entry whose value has '((t, wpos, destroyed_fs, best_score), some_action)'
        # That means the child is the dictionary key, the value is (parent, action).
        # We'll pick the first that matches. That is the action we used to expand from the parent.

        for child_key, val in self.policy.items():
            parent_state, chosen_action = val
            if parent_state == (t, wpos, destroyed_fs, best_score):
                return chosen_action

        # If not found, fallback
        return "terminate"


###############################################################################
# Example usage
###############################################################################
class WizardAgent:
    """
    If you just want a single agent class named `WizardAgent` that
    does the BFS with nonnegative expansions, you can wrap the above BFSWizard inside it.
    """

    def __init__(self, initial_state):
        # Construct the BFS agent
        self.bfs_agent = BFSWizard(initial_state)

    def act(self, state):
        # Just delegate to the BFS agent
        return self.bfs_agent.act(state)
