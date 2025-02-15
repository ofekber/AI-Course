from utils import *

ids = ["208912675", "318159506"]

DESTROY_HORCRUX_REWARD = 2
RESET_REWARD = -2
DEATH_EATER_CATCH_REWARD = -1


def is_passable(cell_char):
    return cell_char.lower() != 'i'


def get_neighbors(r, c, grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    results = []
    for (dr,dc) in [(-1,0),(1,0),(0,-1),(0,1)]:
        rr, cc = r+dr, c+dc
        if 0 <= rr < rows and 0 <= cc < cols:
            if is_passable(grid[rr][cc]):
                results.append((rr,cc))
    return results


class WizardAgent:
    """
    A 'smarter' BFS-based agent that:
      1) Checks collision probability for each tile (1-step look-ahead).
      2) Accepts collisions if net-positive (destroying a Horcrux).
      3) Occasionally resets if heavily cornered.
      4) Can wait near a Horcrux's possible location if the current location is blocked.
      5) Terminates if no better move is feasible.
    """

    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.grid = initial_state["map"]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows > 0 else 0

        self.wizard_names = list(initial_state["wizards"].keys())
        self.horcrux_names = list(initial_state["horcrux"].keys())

        # Death Eater paths for collision prob calculation
        self.death_eater_paths = {
            dname: ddata["path"] for dname, ddata in initial_state["death_eaters"].items()
        }

        # Heuristic parameters (tweak as desired):
        self.collision_threshold = 0.4    # skip tiles with collision prob >= 0.4
        self.cornered_collision_sum = 2.0 # if total collision risk > 2 => reset

    def _death_eater_collision_prob(self, state, cell):
        """
        Estimate the probability that at least one Death Eater
        occupies 'cell' next turn. For each DE, we see where it can move
        (index-1, index, index+1) with equal probability.
        Then use P(not occupied) = Π(1 - p(DE_i on cell)).
        """

        death_eaters = state["death_eaters"]
        p_not_occupied = 1.0

        for dname, ddata in death_eaters.items():
            path = self.death_eater_paths[dname]
            idx = ddata["index"]
            possible_next_indices = [idx]
            if idx > 0:
                possible_next_indices.append(idx-1)
            if idx < len(path)-1:
                possible_next_indices.append(idx+1)

            p_thisDE_on_cell = 0.0
            each_prob = 1.0/len(possible_next_indices)
            for ni in possible_next_indices:
                if path[ni] == cell:
                    p_thisDE_on_cell += each_prob

            p_not_occupied *= (1.0 - p_thisDE_on_cell)

        return 1.0 - p_not_occupied

    def act(self, state):
        turns_left = state["turns_to_go"]
        if turns_left <= 0:
            return "terminate"

        # Possibly check if total collision risk is big => maybe reset
        total_risk = 0.0
        for wname, wdata in state["wizards"].items():
            p_col = self._death_eater_collision_prob(state, wdata["location"])
            total_risk += p_col
        # If we are heavily cornered and have enough turns to try again => reset
        if total_risk > self.cornered_collision_sum and turns_left > 5:
            return "reset"

        # If no horcrux => terminate
        if len(state["horcrux"]) == 0:
            return "terminate"

        actions = []
        did_any_action = False
        wizards_dict = state["wizards"]
        horcruxes_dict = state["horcrux"]

        # Build sets of all current horcrux-locations + their possible-locations
        # so we can try "waiting" if current horcrux location is unreachable.
        horcrux_locs = set()
        possible_horcrux_spots = set()
        for hname, hdata in horcruxes_dict.items():
            horcrux_locs.add(hdata["location"])
            possible_horcrux_spots.update(hdata["possible_locations"])

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

            # 2) BFS to the nearest *reachable* Horcrux location
            #    If BFS fails for all current horcrux-locs, consider BFS to
            #    any *possible-loc* (for waiting/ambush).
            best_target = None
            best_path = None

            def bfs_to_targets(start, targets, collision_check=True):
                """Return the first found BFS path from start to any cell in 'targets' (a set)."""
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
                                # If tile is one of the 'targets' (i.e. has a horcrux),
                                # we might accept collisions if net > 0 => basically p_col < 2.0 => always true
                                # If it's a normal tile => require p_col < threshold
                                if nbr not in horcrux_locs and p_col >= self.collision_threshold:
                                    continue
                            visited[nbr] = cur
                            queue.append(nbr)

                if not found:
                    return None
                # reconstruct path
                path = []
                back = found
                while back is not None:
                    path.append(back)
                    back = visited[back]
                path.reverse()
                return path

            # First try BFS to the actual horcrux-locs
            path_hx = bfs_to_targets(wloc, horcrux_locs, collision_check=True)
            if path_hx:
                best_path = path_hx
                best_target = path_hx[-1]
            else:
                # If that fails, BFS to any possible-loc to "wait" for a teleport
                path_possible = bfs_to_targets(wloc, possible_horcrux_spots, collision_check=True)
                if path_possible:
                    best_path = path_possible
                    best_target = path_possible[-1]
                else:
                    # no BFS success => might do "wait"
                    pass

            if best_path and len(best_path) > 1:
                # Move one step
                next_step = best_path[1]
                actions.append(("move", wname, next_step))
                did_any_action = True

        # If we haven't done anything (no BFS or no destroy)
        if not did_any_action:
            # We can "wait" or "terminate"
            # Possibly if we have turns to spare, we wait to see if a horcrux teleports
            if turns_left > 1:
                wait_actions = []
                for wname in wizards_dict.keys():
                    wait_actions.append(("wait", wname))
                return tuple(wait_actions)
            else:
                return "terminate"

        return tuple(actions)


###############################################################################
# Fixed "OptimalWizardAgent"
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
                "possible_locs": tuple(hdata["possible_locations"]),  # store as tuple
                "prob_change": hdata["prob_change_location"]
            }

        self.policy = {}
        self.value_function = {}

        # Build & solve the MDP offline:
        self.build_and_solve_mdp()

    def build_and_solve_mdp(self):
        # 1) Gather all passable cells for wizard positions
        passable_cells = []
        rows = len(self.grid)
        cols = len(self.grid[ 0 ]) if rows > 0 else 0
        for r in range(rows):
            for c in range(cols):
                if is_passable(self.grid[r][c]):
                    passable_cells.append((r, c))
        passable_cells = tuple(passable_cells)

        # 2) All possible combos of wizard positions:
        #    We'll do a simple Cartesian product of passable_cells for each wizard.
        def all_wizard_positions(cells, num_wizards):
            if num_wizards == 0:
                return [()]  # no wizards => empty tuple
            # For small grid, this might be okay. For large, it could blow up.
            # This is just demonstration code.
            from itertools import product
            return list(product(cells, repeat=num_wizards))

        wizard_pos_combos = all_wizard_positions(passable_cells, len(self.wizard_names))

        # 3) All possible combos of Death Eater indices:
        #    Each DE can be from 0..(len(path)-1).
        de_idx_combos = []
        from itertools import product
        de_ranges = [ range(len(self.death_eater_paths[ d ])) for d in self.death_eater_names ]
        de_idx_combos = list(product(*de_ranges))

        # 4) All possible combos of Horcrux positions:
        #    Each horcrux can be in one of its "possible_locs" (the code includes the current location).
        hx_pos_combos = []
        hx_ranges = [self.horcrux_data[h]["possible_locs"] for h in self.horcrux_names]
        hx_pos_combos = list(product(*hx_ranges))

        #
        # We do a backward induction:
        #   - For t=0, we define value = 0 for *all states* because no more turns to act.
        #   - For t in [1 .. turn_limit], compute the best action by enumerating all feasible actions
        #     and summing over next-state transitions (with probability).
        #

        # Initialize t=0
        for wizPos in wizard_pos_combos:
            for deIdx in de_idx_combos:
                for hxPos in hx_pos_combos:
                    state = (0, wizPos, deIdx, hxPos)
                    self.value_function[state] = 0.0
                    self.policy[state] = "terminate"

        # Now fill for t=1..turn_limit
        for t in range(1, self.turn_limit + 1):
            for wizPos in wizard_pos_combos:
                for deIdx in de_idx_combos:
                    for hxPos in hx_pos_combos:
                        state = (t, wizPos, deIdx, hxPos)
                        best_val = float("-inf")
                        best_action = "terminate"

                        # Enumerate feasible actions from this state
                        possible_actions = self.get_feasible_actions(wizPos, hxPos)

                        for action in possible_actions:
                            # Next-state distribution
                            transitions = self.get_transition_distribution(t, wizPos, deIdx, hxPos, action)
                            # Expected value = sum_{next_s}( prob * (reward + V[next_s]) )
                            exp_val = 0.0
                            for (next_s, prob, rew) in transitions:
                                exp_val += prob * (rew + self.value_function[ next_s ])

                            if exp_val > best_val:
                                best_val = exp_val
                                best_action = action

                        self.value_function[state] = best_val
                        self.policy[state] = best_action

    def get_feasible_actions(self, wiz_positions, horcrux_positions):
        """
        Return the *joint* actions for all wizards simultaneously.
          - We always allow "terminate" and "reset" as single actions (they are global).
          - Otherwise, for each wizard we build a list of sub-actions:
              * "destroy" (if on horcrux),
              * "move" to each neighbor,
              * "wait"
          Then we do a Cartesian product across wizards to yield all possible *joint* actions.
        """
        # If 0 wizards => only "terminate" or "reset"
        if len(wiz_positions) == 0:
            return ["terminate", "reset"]

        import itertools

        # 1) Start with these "single" environment-level actions
        actions = ["terminate", "reset"]

        # 2) For each wizard, build a list of possible sub-actions
        wizard_subactions = []
        for w_idx, wloc in enumerate(wiz_positions):
            wname = self.wizard_names[w_idx]
            subacts = []

            # (a) "destroy" if on a horcrux
            for h_idx, hx_loc in enumerate(horcrux_positions):
                if wloc == hx_loc:
                    hname = self.horcrux_names[h_idx]
                    subacts.append(("destroy", wname, hname))

            # (b) "move" to each neighbor
            neighbors = get_neighbors(wloc[ 0 ], wloc[ 1 ], self.grid)
            for nbr in neighbors:
                subacts.append(("move", wname, nbr))

            # (c) "wait"
            subacts.append(("wait", wname))

            wizard_subactions.append(subacts)

        # 3) Do a cartesian product across all wizards’ subactions
        #    e.g. for 2 wizards, subactions for wizard1 × subactions for wizard2
        #    to form a single "joint action" = tuple( (subact1), (subact2), ... ).
        joint_subacts = itertools.product(*wizard_subactions)
        for joint in joint_subacts:
            # 'joint' is a tuple of subactions, one per wizard
            actions.append(joint)

        return actions

    def get_transition_distribution(self, t, wizPos, deIdx, hxPos, action):
        """
        Compute all (next_state, probability, immediate_reward) for taking 'action'
        in state (t, wizPos, deIdx, hxPos).
        We'll do backward induction, so next_state is (t-1, ...).
        """
        if action == "terminate":
            # No more changes, we go to t-0 with same positions. No reward gained/lost here.
            # Usually you'd say game ends, so next_s might not matter. We'll store next_t=0.
            next_s = (0, wizPos, deIdx, hxPos)
            return [(next_s, 1.0, 0.0)]

        if action == "reset":
            # We reset wizard positions, death eaters, horcrux positions to their *initial*,
            # but reduce time by 1. We get -2 points for a reset.
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

        # Otherwise, action is a tuple of "atomic" wizard sub-actions (like ("move", w1, (r,c))).
        newWizPos_list = list(wizPos)
        immediate_reward = 0.0

        for atom in action:
            atype = atom[0]
            if atype == "move":
                _, wname, target_coord = atom
                w_idx = self.wizard_names.index(wname)
                # Basic check for adjacency
                (r, c) = wizPos[w_idx]
                (tr, tc) = target_coord
                if abs(tr - r) + abs(tc - c) == 1:
                    if is_passable(self.grid[tr][tc]):
                        newWizPos_list[w_idx] = (tr, tc)

            elif atype == "destroy":
                # +2 immediate
                immediate_reward += DESTROY_HORCRUX_REWARD

            elif atype == "wait":
                pass  # do nothing

        newWizPos_tuple = tuple(newWizPos_list)

        # Death Eaters: each can move -1, 0, +1 index in path
        # Let's gather all combinations
        from itertools import product
        death_eater_next_indices_lists = []
        for d_i, dname in enumerate(self.death_eater_names):
            path = self.death_eater_paths[dname]
            cur_idx = deIdx[d_i]
            possible_next = [cur_idx]  # stay
            if cur_idx > 0:
                possible_next.append(cur_idx - 1)
            if cur_idx < len(path) - 1:
                possible_next.append(cur_idx + 1)
            # Probability = 1/len(possible_next)
            death_eater_next_indices_lists.append(possible_next)

        # Horcruxes: each can remain or teleport with prob = prob_change
        # We'll build a distribution for each horcrux individually, then do product.
        def horcrux_distribution(hname, old_loc):
            data = self.horcrux_data[hname]
            p_change = data["prob_change"]
            locs = data["possible_locs"]

            # We'll accumulate (loc -> prob)
            from collections import defaultdict
            dist_map = defaultdict(float)
            if len(locs) > 0:
                # with prob p_change, pick any loc in locs uniformly
                each_p = p_change / len(locs)
                for L in locs:
                    dist_map[L] += each_p
            # with prob (1-p_change) => stay put
            dist_map[old_loc] += (1.0 - p_change)

            return list(dist_map.items())  # list of (loc, prob)

        # 1) Death Eater combos
        transitions = []
        all_de_combos = list(product(*death_eater_next_indices_lists))  # cartesian product

        # 2) Horcrux combos
        hx_distributions = [ ]
        for h_i, hname in enumerate(self.horcrux_names):
            old_loc = hxPos[h_i]
            hx_distributions.append(horcrux_distribution(hname, old_loc))

        # We'll do a cartesian product over these per-horcrux distributions
        def cartesian_hx(idx, current_locs, current_prob):
            if idx == len(hx_distributions):
                yield (tuple(current_locs), current_prob)
                return
            for (loc, p) in hx_distributions[ idx ]:
                cartesian_prob = current_prob * p
                cartesian_locs = current_locs + [ loc ]
                yield from cartesian_hx(idx + 1, cartesian_locs, cartesian_prob)

        all_hx_combos = list(cartesian_hx(0, [], 1.0))

        for de_combo in all_de_combos:
            # Probability for this DE combo is product of 1/len(possible_next_i) for i in each DE
            # But we built them as direct product, so let's compute p_de:
            p_de = 1.0
            for d_i, next_idx in enumerate(de_combo):
                path = self.death_eater_paths[self.death_eater_names[d_i]]
                # how many possible next states?
                cur_idx = deIdx[d_i]
                possibilities = [ cur_idx ]
                if cur_idx > 0: possibilities.append(cur_idx - 1)
                if cur_idx < len(path) - 1: possibilities.append(cur_idx + 1)
                # each has 1/len(possibilities)
                p_de *= (1.0 / len(possibilities))

            for (hx_new_locs, p_hx) in all_hx_combos:
                prob = p_de * p_hx
                if prob <= 0.0:
                    continue

                # next time-step
                next_t = t - 1
                next_deIdx = tuple(de_combo)
                next_hxPos = tuple(hx_new_locs)
                next_s = (next_t, newWizPos_tuple, next_deIdx, next_hxPos)

                # collisions => each wizard with each Death Eater location
                r = immediate_reward
                # figure out the actual location of each DE
                for d_i, ndx in enumerate(next_deIdx):
                    dLoc = self.death_eater_paths[ self.death_eater_names[ d_i ] ][ ndx ]
                    for wloc in newWizPos_tuple:
                        if wloc == dLoc:
                            r += DEATH_EATER_CATCH_REWARD

                transitions.append((next_s, prob, r))

        return transitions

    def act(self, state):
        """
        Look up (t, wizardPositions, deathEaterIndices, horcruxPositions) in self.policy,
        return best action. If not found, default to 'terminate'.
        """
        t = state[ "turns_to_go" ]

        # build wizardPositions
        wpos_list = [ ]
        for wname in self.wizard_names:
            wpos_list.append(state[ "wizards" ][ wname ][ "location" ])
        wpos_tuple = tuple(wpos_list)

        # build deathEaterIndices
        deIdx_list = [ ]
        for dname in self.death_eater_names:
            deIdx_list.append(state[ "death_eaters" ][ dname ][ "index" ])
        deIdx_tuple = tuple(deIdx_list)

        # build horcruxPositions
        hxPos_list = [ ]
        for hname in self.horcrux_names:
            hxPos_list.append(state[ "horcrux" ][ hname ][ "location" ])
        hxPos_tuple = tuple(hxPos_list)

        mdp_state = (t, wpos_tuple, deIdx_tuple, hxPos_tuple)
        return self.policy.get(mdp_state, "terminate")
