from collections import deque
from math import inf

# IDs as requested
ids = ["208912675", "318159506"]

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
    cols = len(grid[0]) if rows > 0 else 0
    results = []
    for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            if is_passable(grid[rr][cc]):
                results.append((rr, cc))
    return results


###############################################################################
# BFS-based wizard agent
###############################################################################
class BFSWizard:
    """
    A BFS-based agent that:
      1) Checks collision probability for each tile (1-step look-ahead).
      2) Accepts collisions if net-positive (destroying a Horcrux).
      3) Occasionally resets if heavily cornered.
      4) Can wait near a Horcrux's possible location if blocked.
      5) Terminates if no better move is feasible.
    """

    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.grid = initial_state[ "map" ]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows > 0 else 0

        self.wizard_names = list(initial_state["wizards"].keys())
        self.horcrux_names = list(initial_state["horcrux"].keys())

        # Death Eater paths for collision probability
        self.death_eater_paths = {
            dname: ddata["path"] for dname, ddata in initial_state["death_eaters"].items()
        }

        # Heuristic parameters
        self.collision_threshold = 0.7  # skip tiles with collision prob >= 0.4
        self.cornered_collision_sum = 2.0  # if total collision risk > 2 => reset

    def _death_eater_collision_prob(self, state, cell):
        """
        Estimate the probability that at least one Death Eater
        occupies 'cell' next turn. For each DE, we see where it can move
        (index-1, index, index+1) with equal probability.
        Then use P(not occupied) = product(1 - p(DE_i on cell)).
        """
        death_eaters = state["death_eaters"]
        p_not_occupied = 1.0

        for dname, ddata in death_eaters.items():
            path = self.death_eater_paths[dname]
            idx = ddata["index"]
            possible_next_indices = [idx]
            if idx > 0:
                possible_next_indices.append(idx - 1)
            if idx < len(path) - 1:
                possible_next_indices.append(idx + 1)

            p_thisDE_on_cell = 0.0
            each_prob = 1.0 / len(possible_next_indices)
            for ni in possible_next_indices:
                if path[ni] == cell:
                    p_thisDE_on_cell += each_prob

            p_not_occupied *= (1.0 - p_thisDE_on_cell)

        return 1.0 - p_not_occupied

    def _bfs_to_targets(self, state, start, targets, collision_check, horcrux_locs, bfs_cache):
        """
        Return a BFS path from 'start' to the nearest cell in 'targets',
        or None if not reachable. We skip neighbors if collision probability >= threshold
        (unless the neighbor is itself a horcrux location).
        We use 'bfs_cache' to avoid repeated BFS calculations in the same turn.
        """

        # BFS cache key
        de_indices = tuple(sorted((dn, dd["index"]) for dn, dd in state["death_eaters"].items()))
        cache_key = (start, frozenset(targets), collision_check, de_indices)

        if cache_key in bfs_cache:
            return bfs_cache[cache_key]

        visited = {start: None}
        queue = deque([start])
        found = None

        while queue and not found:
            cur = queue.popleft()
            if cur in targets:
                found = cur
                break

            for nbr in get_neighbors(cur[0], cur[1], self.grid):
                if nbr not in visited:
                    if collision_check:
                        p_col = self._death_eater_collision_prob(state, nbr)
                        # If tile is not a horcrux-loc, skip if p_col >= threshold
                        if (nbr not in horcrux_locs) and (p_col >= self.collision_threshold):
                            continue
                    visited[nbr] = cur
                    queue.append(nbr)

        if found is None:
            bfs_cache[cache_key] = None
            return None

        # Reconstruct path
        path = []
        back = found
        while back is not None:
            path.append(back)
            back = visited[back]
        path.reverse()

        bfs_cache[cache_key] = path
        return path

    def act(self, state):
        turns_left = state["turns_to_go"]
        if turns_left <= 0:
            return "terminate"

        # Check total collision risk => maybe reset
        total_risk = 0.0
        for wname, wdata in state["wizards"].items():
            p_col = self._death_eater_collision_prob(state, wdata["location"])
            total_risk += p_col
        if total_risk > self.cornered_collision_sum and turns_left > 5:
            return "reset"

        # If no horcrux => terminate
        if len(state["horcrux"]) == 0:
            return "terminate"

        actions = []
        did_any_action = False
        wizards_dict = state["wizards"]
        horcruxes_dict = state["horcrux"]

        # Build sets of current horcrux-locs + possible-locs
        horcrux_locs = set()
        possible_horcrux_spots = set()
        for hname, hdata in horcruxes_dict.items():
            horcrux_locs.add(hdata["location"])
            possible_horcrux_spots.update(hdata["possible_locations"])

        # Cache BFS results for this turn
        bfs_cache = {}

        # 3) For each wizard, choose exactly one action
        for wname, wdata in wizards_dict.items():
            wloc = wdata["location"]

            # (a) If wizard is on a horcrux => destroy
            destroyed = False
            for hname, hdata in horcruxes_dict.items():
                if wloc == hdata["location"]:
                    actions.append(("destroy", wname, hname))
                    destroyed = True
                    break

            if destroyed:
                # Move to next wizard (no BFS for this one)
                continue

            # (b) Try BFS to nearest *reachable* horcrux location
            path_hx = self._bfs_to_targets(
                state, wloc, horcrux_locs, collision_check=True,
                horcrux_locs=horcrux_locs, bfs_cache=bfs_cache
            )

            best_path = path_hx
            # If no BFS path to an actual horcrux, try possible-locs
            if not best_path:
                path_possible = self._bfs_to_targets(
                    state, wloc, possible_horcrux_spots, collision_check=True,
                    horcrux_locs=horcrux_locs, bfs_cache=bfs_cache
                )
                if path_possible:
                    best_path = path_possible

            # (c) If we found a BFS path, move one step
            if best_path and len(best_path) > 1:
                next_step = best_path[1]
                actions.append(("move", wname, next_step))
            else:
                # (d) Otherwise, wizard waits
                # If you want global logic here, you can do so,
                # but at least each wizard gets an action.
                actions.append(("wait", wname))

        # Return all wizard actions as a tuple
        return tuple(actions)


###############################################################################
# Original OptimalWizardAgent
###############################################################################
class OptimalWizardAgent:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.grid = initial_state["map"]

        self.wizard_names = list(initial_state["wizards"].keys())
        self.death_eater_names = list(initial_state["death_eaters"].keys())
        self.horcrux_names = list(initial_state["horcrux"].keys())

        self.turn_limit = initial_state["turns_to_go"]

        # Save Death Eater paths
        self.death_eater_paths = {}
        for dname, ddata in initial_state["death_eaters"].items():
            self.death_eater_paths[dname] = ddata["path"]

        # Save Horcrux possible-locations & teleport probability
        self.horcrux_data = {}
        for hname, hdata in initial_state["horcrux"].items():
            self.horcrux_data[hname] = {
                "location": hdata["location"],
                "possible_locs": tuple(hdata["possible_locations"]),
                "prob_change": hdata["prob_change_location"]
            }

        self.policy = {}
        self.value_function = {}

        # Build & solve the MDP
        self.build_and_solve_mdp()

    def build_and_solve_mdp(self):
        # 1) Gather passable cells for wizard positions
        passable_cells = []
        rows = len(self.grid)
        cols = len(self.grid[0]) if rows > 0 else 0
        for r in range(rows):
            for c in range(cols):
                if is_passable(self.grid[r][c]):
                    passable_cells.append((r, c))
        passable_cells = tuple(passable_cells)

        # 2) All combos of wizard positions
        def all_wizard_positions(cells, num_wizards):
            if num_wizards == 0:
                return [()]
            from itertools import product
            return list(product(cells, repeat=num_wizards))

        wizard_pos_combos = all_wizard_positions(passable_cells, len(self.wizard_names))

        # 3) All combos of Death Eater indices
        from itertools import product
        de_ranges = [range(len(self.death_eater_paths[d])) for d in self.death_eater_names]
        de_idx_combos = list(product(*de_ranges))

        # 4) All combos of Horcrux positions
        hx_ranges = [self.horcrux_data[h]["possible_locs"] for h in self.horcrux_names]
        hx_pos_combos = list(product(*hx_ranges))

        # For t=0, value = 0, action = "terminate"
        for wizPos in wizard_pos_combos:
            for deIdx in de_idx_combos:
                for hxPos in hx_pos_combos:
                    state = (0, wizPos, deIdx, hxPos)
                    self.value_function[state] = 0.0
                    self.policy[state] = "terminate"

        # A cache for transitions so we don't recalc them for the same state-action
        self.transition_cache = {}

        # Now fill t=1..turn_limit (backward induction)
        for t in range(1, self.turn_limit + 1):
            for wizPos in wizard_pos_combos:
                for deIdx in de_idx_combos:
                    for hxPos in hx_pos_combos:
                        state = (t, wizPos, deIdx, hxPos)
                        best_val = float("-inf")
                        best_action = "terminate"

                        # Feasible actions
                        possible_actions = self.get_feasible_actions(wizPos, hxPos)

                        for action in possible_actions:
                            transitions = self.get_transition_distribution_cached(
                                t, wizPos, deIdx, hxPos, action
                            )
                            # expected value
                            exp_val = 0.0
                            for (next_s, prob, rew) in transitions:
                                exp_val += prob * (rew + self.value_function[next_s])

                            if exp_val > best_val:
                                best_val = exp_val
                                best_action = action

                        self.value_function[state] = best_val
                        self.policy[state] = best_action

    def get_feasible_actions(self, wiz_positions, horcrux_positions):
        """
        Return *joint* actions for all wizards:
          - "terminate" or "reset" as single global actions
          - Otherwise, Cartesian product of sub-actions for each wizard:
             * "destroy" (if on horcrux)
             * "move" to neighbor
             * "wait"
        """
        if len(wiz_positions) == 0:
            return ["terminate", "reset"]

        import itertools
        actions = ["terminate", "reset"]

        # Build sub-actions for each wizard
        wizard_subactions = []
        for w_idx, wloc in enumerate(wiz_positions):
            wname = self.wizard_names[w_idx]
            subacts = []

            # (a) 'destroy' if on a horcrux
            for h_idx, hx_loc in enumerate(horcrux_positions):
                if wloc == hx_loc:
                    hname = self.horcrux_names[h_idx]
                    subacts.append(("destroy", wname, hname))

            # (b) 'move'
            neighbors = get_neighbors(wloc[0], wloc[1], self.grid)
            for nbr in neighbors:
                subacts.append(("move", wname, nbr))

            # (c) 'wait'
            subacts.append(("wait", wname))

            wizard_subactions.append(subacts)

        # Cartesian product
        for joint in itertools.product(*wizard_subactions):
            actions.append(joint)

        return actions

    def get_transition_distribution_cached(self, t, wizPos, deIdx, hxPos, action):
        key = (t, wizPos, deIdx, hxPos, action)
        if key in self.transition_cache:
            return self.transition_cache[key]

        trans = self.get_transition_distribution(t, wizPos, deIdx, hxPos, action)
        self.transition_cache[key] = trans
        return trans

    def get_transition_distribution(self, t, wizPos, deIdx, hxPos, action):
        """
        Return list of (next_state, probability, immediate_reward).
        next_state = (t-1, newWizPos, newDeIdx, newHxPos)
        """
        if action == "terminate":
            next_s = (0, wizPos, deIdx, hxPos)
            return [(next_s, 1.0, 0.0)]

        if action == "reset":
            newWizPos = []
            for wname in self.wizard_names:
                newWizPos.append(self.initial_state["wizards"][wname]["location"])
            newWizPos = tuple(newWizPos)

            newDeIdx = []
            for dname in self.death_eater_names:
                newDeIdx.append(self.initial_state["death_eaters"][dname]["index"])
            newDeIdx = tuple(newDeIdx)

            newHxPos = []
            for hname in self.horcrux_names:
                newHxPos.append(self.initial_state["horcrux"][hname]["location"])
            newHxPos = tuple(newHxPos)

            next_s = (t - 1, newWizPos, newDeIdx, newHxPos)
            return [(next_s, 1.0, RESET_REWARD)]

        # Otherwise, a joint action for wizards
        if isinstance(action, tuple) and len(action) > 0 and isinstance(action[0], str):
            joint_actions = [action]
        else:
            joint_actions = list(action)

        newWizPos_list = list(wizPos)
        immediate_reward = 0.0

        # Process each sub-action
        for atom in joint_actions:
            atype = atom[0]
            if atype == "move":
                _, wname, target_coord = atom
                w_idx = self.wizard_names.index(wname)
                (r, c) = wizPos[w_idx]
                (tr, tc) = target_coord
                if abs(tr - r) + abs(tc - c) == 1 and is_passable(self.grid[tr][tc]):
                    newWizPos_list[w_idx] = (tr, tc)

            elif atype == "destroy":
                immediate_reward += DESTROY_HORCRUX_REWARD

            elif atype == "wait":
                pass  # no effect

        newWizPos_tuple = tuple(newWizPos_list)

        from itertools import product
        death_eater_next_indices_lists = []
        for d_i, dname in enumerate(self.death_eater_names):
            path = self.death_eater_paths[dname]
            cur_idx = deIdx[d_i]
            possible_next = [cur_idx]
            if cur_idx > 0:
                possible_next.append(cur_idx - 1)
            if cur_idx < len(path) - 1:
                possible_next.append(cur_idx + 1)
            death_eater_next_indices_lists.append(possible_next)

        # Horcrux distribution: remain or teleport
        hx_distributions = []
        for h_i, hname in enumerate(self.horcrux_names):
            old_loc = hxPos[h_i]
            hx_distributions.append(self._horcrux_distribution(hname, old_loc))

        transitions = []

        all_de_combos = list(product(*death_eater_next_indices_lists))

        def cartesian_hx(idx, current_locs, current_prob):
            if idx == len(hx_distributions):
                yield (tuple(current_locs), current_prob)
                return
            for (loc, p) in hx_distributions[idx]:
                yield from cartesian_hx(idx + 1, current_locs + [loc], current_prob * p)

        all_hx_combos = list(cartesian_hx(0, [], 1.0))

        for de_combo in all_de_combos:
            # Probability for this DE combo
            p_de = 1.0
            for d_i, next_idx in enumerate(de_combo):
                dname = self.death_eater_names[d_i]
                path = self.death_eater_paths[dname]
                cur_idx = deIdx[d_i]
                possible_next = [cur_idx]
                if cur_idx > 0:
                    possible_next.append(cur_idx - 1)
                if cur_idx < len(path) - 1:
                    possible_next.append(cur_idx + 1)
                p_de *= (1.0 / len(possible_next))

            for (hx_new_locs, p_hx) in all_hx_combos:
                prob = p_de * p_hx
                if prob <= 0.0:
                    continue

                next_t = t - 1
                next_deIdx = tuple(de_combo)
                next_hxPos = tuple(hx_new_locs)
                next_s = (next_t, newWizPos_tuple, next_deIdx, next_hxPos)

                # Check collisions: wizard & DE
                r = immediate_reward
                for d_i, ndx in enumerate(next_deIdx):
                    dLoc = self.death_eater_paths[self.death_eater_names[d_i]][ndx]
                    for wloc in newWizPos_tuple:
                        if wloc == dLoc:
                            r += DEATH_EATER_CATCH_REWARD

                transitions.append((next_s, prob, r))

        return transitions

    def _horcrux_distribution(self, hname, old_loc):
        """
        Return list of (new_loc, p). With prob p_change, pick any possible loc
        uniformly. With prob (1 - p_change), stay put.
        """
        data = self.horcrux_data[hname]
        p_change = data["prob_change"]
        locs = data["possible_locs"]

        from collections import defaultdict
        dist_map = defaultdict(float)
        if len(locs) > 0:
            each_p = p_change / len(locs)
            for L in locs:
                dist_map[L] += each_p

        dist_map[old_loc] += (1.0 - p_change)
        return list(dist_map.items())

    def act(self, state):
        """
        Look up (t, wizardPositions, deathEaterIndices, horcruxPositions) in self.policy,
        return best action. If not found, default to "terminate".
        """
        t = state["turns_to_go"]

        # Wizard positions
        wpos_list = []
        for wname in self.wizard_names:
            wpos_list.append(state["wizards"][wname]["location"])
        wpos_tuple = tuple(wpos_list)

        # Death Eater indices
        deIdx_list = []
        for dname in self.death_eater_names:
            deIdx_list.append(state["death_eaters"][dname]["index"])
        deIdx_tuple = tuple(deIdx_list)

        # Horcrux positions
        hxPos_list = []
        for hname in self.horcrux_names:
            hxPos_list.append(state["horcrux"][hname]["location"])
        hxPos_tuple = tuple(hxPos_list)

        mdp_state = (t, wpos_tuple, deIdx_tuple, hxPos_tuple)
        return self.policy.get(mdp_state, "terminate")


###############################################################################
# WizardAgent:
###############################################################################
def _check_state_space_size(state):
    """
    Compute the theoretical number of states |S|.
    Formula: (turns_to_go + 1) *
             (#passable_cells ^ (#wizards)) *
             (product of each Death Eater path length) *
             (product of each Horcrux possible_locs size).
    """
    # 1) Get passable cell count
    grid = state["map"]
    passable_count = sum(
        1 for row in grid for c in row
        if is_passable(c)
    )

    # 2) #wizards => passable_count^(#wizards)
    wizards = state["wizards"]
    num_wizards = len(wizards)
    wizard_factor = passable_count ** num_wizards if num_wizards > 0 else 1

    # 3) product of DE path lengths
    death_eaters = state["death_eaters"]
    de_factor = 1
    for dname, ddata in death_eaters.items():
        de_factor *= len(ddata["path"])

    # 4) product of horcrux possible locs
    horcruxes = state["horcrux"]
    hx_factor = 1
    for hname, hdata in horcruxes.items():
        hx_factor *= len(hdata["possible_locations"])

    # 5) multiply by (turns_to_go + 1)
    T = state["turns_to_go"]
    total_states = (T + 1) * wizard_factor * de_factor * hx_factor
    return total_states


class WizardAgent(OptimalWizardAgent):
    def __init__(self, initial_state):

        # We'll store a BFS fallback agent too
        self.bfs_fallback = BFSWizard(initial_state)

        # Decide if we will skip MDP entirely based on |S|.
        self.skip_mdp = _check_state_space_size(initial_state) > 1000000

        # Let the parent init do basic things, but note that build_and_solve_mdp()
        # is overridden below, so we won't run the parent's version of it fully.
        super().__init__(initial_state)

    def build_and_solve_mdp(self):

        if self.skip_mdp:
            # Skip building the MDP altogether
            return
        # Otherwise, do the parent's logic to build & solve.
        # We can either copy/paste the parent's code, or call super().
        # If you want the parent's EXACT method, just do:
        super().build_and_solve_mdp()

    def act(self, state):
        """
        Looks up (t, wizPos, deIdx, hxPos) in self.policy. If missing, fallback to BFS.
        """
        t = state["turns_to_go"]

        # Wizard positions
        wpos_list = []
        for wname in self.wizard_names:
            wpos_list.append(state["wizards"][wname]["location"])
        wpos_tuple = tuple(wpos_list)

        # Death Eater indices
        deIdx_list = []
        for dname in self.death_eater_names:
            deIdx_list.append(state["death_eaters"][dname]["index"])
        deIdx_tuple = tuple(deIdx_list)

        # Horcrux positions
        hxPos_list = []
        for hname in self.horcrux_names:
            hxPos_list.append(state["horcrux"][hname]["location"])
        hxPos_tuple = tuple(hxPos_list)

        mdp_state = (t, wpos_tuple, deIdx_tuple, hxPos_tuple)

        # Look in parent's policy dictionary
        action = self.policy.get(mdp_state, None)
        if action is not None:
            return action

        # Otherwise, BFS fallback:
        return self.bfs_fallback.act(state)
