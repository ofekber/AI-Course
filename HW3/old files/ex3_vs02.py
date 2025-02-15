from utils import *

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

def cartesian_product(*args):
    """
    Generate the Cartesian product of input lists.
    Yields tuples representing the Cartesian product.
    """
    if not args:
        yield ()
        return

    first, *rest = args
    for item in first:
        for items in cartesian_product(*rest):
            yield (item,) + items

###############################################################################
# A BFS-based WizardAgent (renamed to BFSWizardAgent for reference)
###############################################################################
class BFSWizardAgent:
    """
    A 'smarter' BFS-based agent that:
      1) Checks collision probability for each tile (1-step look-ahead).
      2) Accepts collisions if net-positive (destroying a Horcrux).
      3) Occasionally resets if heavily cornered.
      4) Can wait near a Horcrux's possible location if blocked.
      5) Terminates if no better move is feasible.
    """

    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.grid = initial_state["map"]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows > 0 else 0

        self.wizard_names = list(initial_state["wizards"].keys())
        self.horcrux_names = list(initial_state["horcrux"].keys())

        # Death Eater paths for collision probability
        self.death_eater_paths = {
            dname: ddata["path"] for dname, ddata in initial_state["death_eaters"].items()
        }

        # Heuristic parameters
        self.collision_threshold = 0.4    # skip tiles with collision prob >= 0.4
        self.cornered_collision_sum = 2.0 # if total collision risk > 2 => reset

    def _death_eater_collision_prob(self, state, cell):
        """
        Estimate the probability that at least one Death Eater occupies 'cell' next turn.
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
        Return a BFS path from 'start' to the nearest cell in 'targets', or None if not reachable.
        We skip neighbors if collision probability >= threshold (unless it's a horcrux location).
        We use 'bfs_cache' to avoid repeated BFS in the same turn.
        """

        # BFS cache key
        de_indices = tuple(sorted((dn, dd["index"]) for dn, dd in state["death_eaters"].items()))
        cache_key = (start, frozenset(targets), collision_check, de_indices)
        if cache_key in bfs_cache:
            return bfs_cache[cache_key]

        visited = {start: None}
        queue = FIFOQueue(None, [start])
        found = None

        while len(queue) > 0 and not found:
            cur = queue.pop()
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

        # For each wizard:
        for wname, wdata in wizards_dict.items():
            wloc = wdata["location"]

            # 1) If wizard on a horcrux => destroy
            is_destroyed = False
            for hname, hdata in horcruxes_dict.items():
                if wloc == hdata["location"]:
                    actions.append(("destroy", wname, hname))
                    did_any_action = True
                    is_destroyed = True
                    break
            if is_destroyed:
                continue

            # 2) BFS to the nearest *reachable* horcrux location
            path_hx = self._bfs_to_targets(
                state, wloc, horcrux_locs, collision_check=True,
                horcrux_locs=horcrux_locs, bfs_cache=bfs_cache
            )
            best_path = None
            if path_hx:
                best_path = path_hx
            else:
                # If that fails, BFS to possible-locs
                path_possible = self._bfs_to_targets(
                    state, wloc, possible_horcrux_spots, collision_check=True,
                    horcrux_locs=horcrux_locs, bfs_cache=bfs_cache
                )
                if path_possible:
                    best_path = path_possible

            # If BFS succeeded, move one step
            if best_path and len(best_path) > 1:
                next_step = best_path[1]
                actions.append(("move", wname, next_step))
                did_any_action = True

        # If we haven't done anything
        if not did_any_action:
            # Possibly we wait or terminate
            if turns_left > 1:
                wait_actions = []
                for wname in wizards_dict.keys():
                    wait_actions.append(("wait", wname))
                return tuple(wait_actions)
            else:
                return "terminate"

        return tuple(actions)

###############################################################################
# Original OptimalWizardAgent (unchanged)
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

        # Build & solve the MDP (full iteration)
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
            return list(cartesian_product(cells, *([cells] * (num_wizards - 1))))

        wizard_pos_combos = all_wizard_positions(passable_cells, len(self.wizard_names))

        # 3) All combos of Death Eater indices
        de_ranges = [range(len(self.death_eater_paths[d])) for d in self.death_eater_names]
        de_idx_combos = list(cartesian_product(*de_ranges))

        # 4) All combos of Horcrux positions
        hx_ranges = [self.horcrux_data[h]["possible_locs"] for h in self.horcrux_names]
        hx_pos_combos = list(cartesian_product(*hx_ranges))

        # For t=0, value = 0, action = "terminate"
        for wizPos in wizard_pos_combos:
            for deIdx in de_idx_combos:
                for hxPos in hx_pos_combos:
                    state = (0, wizPos, deIdx, hxPos)
                    self.value_function[state] = 0.0
                    self.policy[state] = "terminate"

        # A cache for transitions so we don't recalc them for the same (state, action)
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
             * "move" to a neighbor
             * "wait"
        """
        if len(wiz_positions) == 0:
            return ["terminate", "reset"]

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

            # (b) 'move' to neighbors
            neighbors = get_neighbors(wloc[0], wloc[1], self.grid)
            for nbr in neighbors:
                subacts.append(("move", wname, nbr))

            # (c) 'wait'
            subacts.append(("wait", wname))

            wizard_subactions.append(subacts)

        # All joint combinations
        for joint in cartesian_product(*wizard_subactions):
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
        next_state = (t-1, newWizPos, newDeIdx, newHxPos).
        """
        # If "terminate"
        if action == "terminate":
            next_s = (0, wizPos, deIdx, hxPos)
            return [(next_s, 1.0, 0.0)]

        # If "reset"
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

        # Otherwise, a joint action
        if isinstance(action, tuple) and len(action) > 0 and isinstance(action[0], str):
            # single sub-action
            joint_actions = [action]
        else:
            # multiple wizard sub-actions
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
                # valid move if neighbor is passable
                if abs(tr - r) + abs(tc - c) == 1 and is_passable(self.grid[tr][tc]):
                    newWizPos_list[w_idx] = (tr, tc)

            elif atype == "destroy":
                immediate_reward += DESTROY_HORCRUX_REWARD

            elif atype == "wait":
                pass  # do nothing

        newWizPos_tuple = tuple(newWizPos_list)

        # Death Eaters: each can move -1, 0, or +1 with equal probability
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

        # All combos of next DE indices
        all_de_combos = list(cartesian_product(*death_eater_next_indices_lists))

        # All combos of new Horcrux positions
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
        Return list of (new_loc, p). With prob p_change, pick any possible loc uniformly.
        With prob (1 - p_change), stay put.
        """
        data = self.horcrux_data[hname]
        p_change = data["prob_change"]
        locs = data["possible_locs"]

        dist_map = {}
        if len(locs) > 0:
            each_p = p_change / len(locs)
            for L in locs:
                if L not in dist_map:
                    dist_map[L] = 0.0
                dist_map[L] += each_p

        if old_loc not in dist_map:
            dist_map[old_loc] = 0.0
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
# New WizardAgent that inherits from OptimalWizardAgent
# and allows limiting the value-iteration depth
###############################################################################
class WizardAgent(OptimalWizardAgent):
    def __init__(self, initial_state, iteration_depth = None):
        """
        :param initial_state: The initial state dictionary (same format as before).
        :param iteration_depth: Maximum depth of backward iteration to perform.
                               If None or >= turn_limit, do full iteration.
        """
        self.iteration_depth = iteration_depth
        super().__init__(initial_state)

    def build_and_solve_mdp(self):
        """
        Override the parent's MDP-building method to allow partial (limited-depth)
        backward induction. If iteration_depth is None or large, we default to full
        iteration as in the parent.
        """
        # If no limit or limit exceeds total turns, use the full parent's logic:
        if self.iteration_depth is None or self.iteration_depth >= self.turn_limit:
            return super().build_and_solve_mdp()

        # Otherwise, replicate parent's logic, but restrict the time loop to iteration_depth.
        # 1) Gather passable cells
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
            return list(cartesian_product(cells, *([cells] * (num_wizards - 1))))

        wizard_pos_combos = all_wizard_positions(passable_cells, len(self.wizard_names))

        # 3) All combos of Death Eater indices
        de_ranges = [range(len(self.death_eater_paths[d])) for d in self.death_eater_names]
        de_idx_combos = list(cartesian_product(*de_ranges))

        # 4) All combos of Horcrux positions
        hx_ranges = [self.horcrux_data[h]["possible_locs"] for h in self.horcrux_names]
        hx_pos_combos = list(cartesian_product(*hx_ranges))

        # For t=0, value = 0, action = "terminate"
        for wizPos in wizard_pos_combos:
            for deIdx in de_idx_combos:
                for hxPos in hx_pos_combos:
                    state = (0, wizPos, deIdx, hxPos)
                    self.value_function[state] = 0.0
                    self.policy[state] = "terminate"

        # A cache for transitions
        self.transition_cache = {}

        # We'll only iterate for t = 1..(iteration_depth), even if turn_limit is bigger.
        limit = min(self.turn_limit, self.iteration_depth)

        for t in range(1, limit + 1):
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

        # Note: For t > iteration_depth, we leave the policy at default ("terminate")
        # or whatever it was set to earlier. So effectively, we only partially solve
        # the MDP up to t = iteration_depth.

