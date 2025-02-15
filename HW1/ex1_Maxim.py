import search
import random
import math
import itertools

ids = ["207433715", "323651281"]


class HarryPotterProblem(search.Problem):
    def __init__(self, initial):
        self.grid = initial['map']
        self.death_eaters = initial['death_eaters']
        self.voldemort_pos = self.find_voldemort()
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.all_horcruxes = initial['horcruxes']
        search.Problem.__init__(self, self.convert_to_state(initial))

    def find_voldemort(self):
        """Find Voldemort's position in the grid."""
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == 'V':
                    return (r, c)

    def convert_to_state(self, initial):
        """Converts the initial input dictionary to a hashable state representation."""
        wizard_positions = tuple((wizard, pos) for wizard, (pos, lives) in initial['wizards'].items())
        horcruxes = tuple(initial['horcruxes'])
        wizard_health = tuple((wizard, lives) for wizard, (pos, lives) in initial['wizards'].items())
        death_eaters_position = tuple((death_eater, pos[0], True, 0) for death_eater, pos in initial['death_eaters'].items())
        return (wizard_positions, horcruxes, wizard_health, death_eaters_position, False, 0)

    def actions(self, state):
        """Generate all possible actions for each wizard."""
        wizard_positions = dict(state[0])
        horcruxes = state[1]
        actions = []

        for wizard, pos in wizard_positions.items():
            valid_moves = self.get_valid_moves(pos, wizard)
            if pos in horcruxes:
                valid_moves.append(("destroy", wizard, self.all_horcruxes.index(pos)))
            if pos == self.voldemort_pos and not horcruxes and wizard == "Harry Potter":
                valid_moves.append(("kill", "Harry Potter"))
            valid_moves.append(("wait", wizard))
            actions.append(valid_moves)


        return itertools.product(*actions)

    def get_valid_moves(self, pos, wizard):
        """Generate valid moves for a given position."""
        r, c = pos
        moves = []
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_pos = (r + dr, c + dc)
            if self.is_passable(new_pos):
                moves.append(("move", wizard, new_pos))
        return moves

    def is_passable(self, pos):
        """Check if a position is passable."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and (self.grid[r][c] == 'P' or self.grid[r][c] == 'V')

    def result(self, state, action):
        """Apply an action to a state and return the resulting state."""

        # print("The action chosen:", action)
        wizard_positions = dict(state[0])
        horcruxes = set(state[1])
        wizard_health = dict(state[2])
        death_eaters = list(state[3])
        did_try_to_kill = False
        destroyed = False

        for single_action in action:
            if len(single_action) == 3:
                action_type, wizard, target = single_action
                if action_type == "move":
                    wizard_positions[wizard] = target
                elif action_type == "destroy" and self.all_horcruxes[target] in horcruxes:
                    horcruxes.remove(self.all_horcruxes[target])
                    destroyed = True
            elif single_action[0] == "kill":
                did_try_to_kill = True

        updated_death_eaters = []
        death_eaters_locations = set()
        for death_eater, pos, move_right, idx in death_eaters:
            if len(self.death_eaters[death_eater]) <= 1:
                death_eaters_locations.add(self.death_eaters[death_eater][0])
                continue
            if move_right:
                if idx < len(self.death_eaters[death_eater]) - 1:
                    idx += 1
                    updated_death_eaters.append((death_eater, self.death_eaters[death_eater][idx], True, idx))
                elif idx == len(self.death_eaters[death_eater]) - 1:
                    idx -= 1
                    updated_death_eaters.append((death_eater, self.death_eaters[death_eater][idx], False, idx))
            else:
                if idx > 0:
                    idx -= 1
                    updated_death_eaters.append((death_eater, self.death_eaters[death_eater][idx], False, idx))
                elif idx == 0:
                    idx += 1
                    updated_death_eaters.append((death_eater, self.death_eaters[death_eater][idx], True, idx))

            death_eaters_locations.add(self.death_eaters[death_eater][idx])

        for wizard in wizard_positions.keys():
            if wizard_positions[wizard] in death_eaters_locations:
                wizard_health[wizard] -= 1
            if wizard != "Harry Potter" and wizard_positions[wizard] == self.voldemort_pos:
                wizard_health[wizard] = 0
            if wizard == "Harry Potter" and wizard_positions[wizard] == self.voldemort_pos and (horcruxes or destroyed):
                wizard_health[wizard] = 0


        new_state = (
            tuple(wizard_positions.items()),
            tuple(horcruxes),
            tuple(wizard_health.items()),
            tuple(updated_death_eaters),
            did_try_to_kill,
            state[5] + 1
        )
        return new_state

    def goal_test(self, state):
        """Return True if all horcruxes are destroyed and Voldemort is killed."""
        if state[5] and not state[1]:
            for wizard, lives in state[2]:
                if lives < 1:
                    return False
            return True
        return False

    def h(self, node):
        """
        Heuristic function: Estimate distance to destroy all horcruxes and reach Voldemort.
        Ensures each wizard targets a unique horcrux.
        """
        wizard_positions = dict(node.state[0])
        horcruxes = set(node.state[1])
        wizard_lives = dict(node.state[2])

        for wizard, lives in wizard_lives.items():
            if lives < 1:
                return 1000000

        harry_position = wizard_positions["Harry Potter"]

        claimed_horcruxes = set()
        total_distance = 0

        for wizard, position in wizard_positions.items():
            closest_horcrux_distance = float("inf")
            target_horcrux = None

            for horcrux in horcruxes:
                if horcrux in claimed_horcruxes:
                    continue

                distance = self.manhattan_distance(position, horcrux)
                if distance < closest_horcrux_distance:
                    closest_horcrux_distance = distance
                    target_horcrux = horcrux

            if target_horcrux is not None:
                claimed_horcruxes.add(target_horcrux)
                total_distance += closest_horcrux_distance

        if len(wizard_lives) > len(horcruxes):
            harry_distance_voldemort = self.manhattan_distance(harry_position, self.voldemort_pos)
        else:
            harry_distance_voldemort = 0
        return total_distance + 100 * len(horcruxes) + 26.3 * node.state[5] + 20 * harry_distance_voldemort


    def manhattan_distance(self, start, end):
        """
        Calculate the shortest path between start and end, considering obstacles ('I').
        Harry can move to 'V' when all horcruxes are destroyed.
        """
        sx, sy = start
        ex, ey = end
        return abs(ex - sx) + abs(ey - sy)


def create_harrypotter_problem(game):
    return HarryPotterProblem(game)

