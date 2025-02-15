import search
import itertools
from itertools import cycle
from utils import *

ids = ["208912675", "318159506"]


class HarryPotterProblem(search.Problem):
    def __init__(self, initial):
        # Step 1: Map initialization
        self.map = initial["map"]

        # Step 2: Initialize wizards
        self.wizards = {
            name: {
                "position": details[0],
                "lives": details[1],
            }
            for name, details in initial["wizards"].items()
        }

        # Step 3: Initialize death eaters
        self.death_eaters = initial["death_eaters"]

        # Step 4: Initialize horcruxes
        self.horcruxes = initial["horcruxes"]  # keep it as a list so duplicates are preserved

        # Step 5: Initialize Voldemort's position
        self.voldemort_position = self._find_voldemort()

        # Step 6: Create the initial state
        initial_state = (
            tuple((name, details["position"], details["lives"]) for name, details in self.wizards.items()),
            tuple((name, path[0]) for name, path in self.death_eaters.items()),  # Tuple of tuples for hashability
            tuple(self.horcruxes),
            False
        )

        # Step 7: Call super constructor
        super().__init__(initial_state)

    def _find_voldemort(self):
        """Find Voldemort's position on the map."""
        for i, row in enumerate(self.map):
            for j, cell in enumerate(row):
                if cell == 'V':
                    return i, j
        return None  # Voldemort not found

    def actions(self, state):
        """
        Generate prioritized actions for all wizards in the current state.
        Now allows multiple wizards to be on the same tile in the same round.
        """
        wizards, death_eaters_positions, remaining_horcruxes, voldemort_dead = state

        # Calculate the next moves for death eaters
        death_eater_moves = self._calculate_death_eater_moves(death_eaters_positions)

        possible_actions = []
        rows, cols = len(self.map), len(self.map[0])  # Map dimensions

        for wizard_name, position, lives in wizards:
            wizard_actions = []
            x, y = position

            # Priority 1: Destroy a horcrux if on it
            for i, hpos in enumerate(remaining_horcruxes):
                if hpos == (x, y):
                    if lives > 1 or (x, y) not in death_eater_moves:
                        wizard_actions.append(("destroy", wizard_name, self.horcruxes.index(hpos)))

            # Priority 2: Kill Voldemort (only Harry, if horcruxes are gone, V not dead, on V's tile)
            if (
                    wizard_name == "Harry Potter"
                    and not voldemort_dead
                    and not remaining_horcruxes
                    and (x, y) == self.voldemort_position
            ):
                wizard_actions.append(("kill", wizard_name))

            # Priority 3: Move actions
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:  # Stay within map bounds
                    if self.map[nx][ny] != "I":  # Can't move onto impassable tiles
                        # Avoid death eater if lives <= 1
                        if lives > 1 or (nx, ny) not in death_eater_moves:
                            # Special Voldemort encounter rules
                            if (nx, ny) == self.voldemort_position:
                                if remaining_horcruxes:
                                    continue  # Avoid V if horcruxes remain
                                elif wizard_name != "Harry Potter":
                                    continue  # Only Harry can approach V
                                elif wizard_name == "Harry Potter" and (x, y) in remaining_horcruxes:
                                    continue  # Harry won't move onto V in same turn last horcrux is destroyed
                                wizard_actions.append(("move", wizard_name, (nx, ny)))
                            else:
                                # Normal move
                                wizard_actions.append(("move", wizard_name, (nx, ny)))

            # Priority 4: Wait action if nothing else is valid
            if not wizard_actions:
                wizard_actions.append(("wait", wizard_name))

            possible_actions.append(wizard_actions)

        # Now simply return the Cartesian product of each wizard's possible actions
        return itertools.product(*possible_actions)

    def _calculate_death_eater_moves(self, death_eaters_positions):
        """
        Calculate the next positions for all death eaters based on their paths.
        :param death_eaters_positions: Current positions of death eaters (tuple of tuples).
        :return: Set of all tiles death eaters might move to.
        """
        next_positions = set()

        # Create oscillating paths for all death eaters if not already initialized
        if not hasattr(self, '_death_eater_cycles'):
            self._death_eater_cycles = {
                name: cycle(path + path[-2::-1])  # Oscillating path: forward + reverse
                for name, path in self.death_eaters.items()
            }

        for name, current_pos in death_eaters_positions:
            # Fetch the next position in the cycle for the current Death Eater
            next_pos = next(self._death_eater_cycles[name])
            next_positions.add(next_pos)

        return next_positions

    def result(self, state, action):
        """
        state: (wizards, death_eaters_positions, remaining_horcruxes, voldemort_dead)
        action: a tuple of (action_type, wizard_name, extra...)
        """
        wizards, death_eaters_positions, remaining_horcruxes, voldemort_dead = state
        wizard_dict = {name: (position, lives) for name, position, lives in wizards}
        new_wizards = {}
        new_voldemort_dead = voldemort_dead

        # Convert tuple -> list for partial update
        horcrux_list = list(remaining_horcruxes)

        # 1) Gather all "destroy" indexes from the entire joint action
        destroy_indices = []
        # We'll store everything else (move, wait, kill) normally
        other_actions = []

        for wizard_action in action:
            action_type, wizard_name, *details = wizard_action
            position, lives = wizard_dict[wizard_name]

            if action_type == "destroy":
                horcrux_index = self.horcruxes[details[0]]
                # Collect the index for later removal
                destroy_indices.append(remaining_horcruxes.index(horcrux_index))

                # Wizard stays in place
                new_wizards[wizard_name] = (position, lives)

            else:
                # We'll handle move/wait/kill as normal
                other_actions.append((wizard_action, position, lives))

        # 2) Remove duplicates, so each Horcrux index can only be destroyed once
        destroy_indices = list(set(destroy_indices))
        # Sort in descending order to avoid index shifting issues
        destroy_indices.sort(reverse=True)

        # 3) Actually remove each Horcrux (if in range)
        for idx in destroy_indices:
            if 0 <= idx < len(horcrux_list):
                del horcrux_list[idx]

        # 4) Handle the other actions (move, wait, kill) after Horcrux removal
        for wizard_action, position, lives in other_actions:
            action_type, wizard_name, *details = wizard_action

            if action_type == "move":
                new_position = details[0]
                new_wizards[wizard_name] = (new_position, lives)

            elif action_type == "wait":
                new_wizards[wizard_name] = (position, lives)

            elif action_type == "kill":
                new_voldemort_dead = True
                new_wizards[wizard_name] = (position, lives)

        # 5) Update death eaters (same as your original code)
        updated_positions = {}
        if not hasattr(self, '_death_eater_cycles'):
            self._death_eater_cycles = {
                name: cycle(self.death_eaters[ name ] + self.death_eaters[ name ][ -2::-1 ])
                for name in self.death_eaters
            }

        for name, _ in death_eaters_positions:
            new_pos = next(self._death_eater_cycles[ name ])
            updated_positions[ name ] = new_pos

        new_death_eater_positions = tuple(updated_positions.items())

        # 6) Resolve wizard-death_eater collisions (same as original)
        for wizard_name, (position, lives) in new_wizards.items():
            if position in updated_positions.values() and lives > 1:
                new_wizards[wizard_name] = (position, lives - 1)

        # 7) Return updated state
        next_state = (
            tuple((name, pos, lives) for name, (pos, lives) in new_wizards.items()),
            new_death_eater_positions,
            tuple(horcrux_list),  # convert back to tuple
            new_voldemort_dead
        )
        return next_state

    def goal_test(self, state):
        """
        Returns True if the state is a goal state, False otherwise.
        A goal state satisfies the following:
        1. All wizards that started the game are alive.
        2. All horcruxes are destroyed.
        3. Voldemort is dead.
        """
        wizards, death_eaters_positions, remaining_horcruxes, voldemort_dead = state

        # 1. Check if all wizards are alive
        for wizard in wizards:
            wizard_name, position, lives = wizard
            if lives <= 0:
                return False

        # 2. Check if all horcruxes are destroyed
        if remaining_horcruxes:
            return False

        # 3. Check if Voldemort is dead
        harry_on_voldemort = False
        for (w_name, pos, _) in wizards:
            if w_name == "Harry Potter" and pos == self.voldemort_position:
                harry_on_voldemort = True
                break

        if harry_on_voldemort and voldemort_dead:
            return True
        elif harry_on_voldemort and not voldemort_dead:
            return False
        else:
            # If Harry is not even on Voldemort's tile, we can't be at a goal.
            return False

    def h(self, node):
        (wizards, _, remaining_horcruxes, voldemort_dead) = node.state

        # If Voldemort is already dead, no cost
        if voldemort_dead:
            return 0

        # Use current heuristic for smaller maps
        total_distance = 0
        horcrux_count = len(remaining_horcruxes)

        # Calculate minimum distances to horcruxes for each wizard
        wizard_to_horcrux = []
        for wizard_name, position, _ in wizards:
            if remaining_horcruxes:
                distances = [self._manhattan_distance(position, h) for h in remaining_horcruxes]
                wizard_to_horcrux.append(min(distances, default=0))  # Nearest horcrux for this wizard
            elif wizard_name == "Harry Potter":
                # Add distance to Voldemort if no horcruxes remain
                total_distance += self._manhattan_distance(position, self.voldemort_position)

        # Minimize the maximum distance among all wizard assignments to horcruxes
        total_distance += max(wizard_to_horcrux, default=0)

        # Depth penalty scaled based on remaining tasks
        depth_penalty = node.depth * (2.3 + horcrux_count)

        # Idle penalty to discourage inactive wizards
        idle_penalty = sum(1 for wizard in wizards if wizard[1] == "wait")

        return total_distance + depth_penalty + idle_penalty

    @staticmethod
    def _manhattan_distance(pos1, pos2):
        """
        Calculate Manhattan distance between two positions.
        :param pos1: First position as a tuple (x, y).
        :param pos2: Second position as a tuple (x, y).
        :return: Manhattan distance between pos1 and pos2.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def create_harrypotter_problem(game):
    return HarryPotterProblem(game)
