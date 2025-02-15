from utils import *
ids = ["208912675", "318159506"]


class GringottsController:
    def __init__(self, map_shape, harry_loc, initial_observations):

        self.map_shape = map_shape
        self.harry_loc = harry_loc

        # Known facts:
        self.dragons = set()
        self.suss = set()
        self.safe = set()  # squares we consider safe
        self.vaults = set()
        self.collected_vaults = set()  # which vaults we've tried "collect"

        # We'll also keep a visited set if we want to prefer exploring new squares
        self.visited = set()

        # Mark the starting location as safe
        self.safe.add(harry_loc)
        self.visited.add(harry_loc)

        # Process the initial observations
        self._update_knowledge(initial_observations)

    def get_next_action(self, observations):

        # 1) Update knowledge
        self._update_knowledge(observations)


        # 2) If we are on a new vault => collect
        if self.harry_loc in self.vaults and self.harry_loc not in self.collected_vaults:
            self.collected_vaults.add(self.harry_loc)
            return ("collect",)  # if it's the hallow vault, game ends

        # 3) Attempt a local move
        neighbors = self._get_neighbors(self.harry_loc)
        random.shuffle(neighbors)  # This will shuffle the list in place

        # 4) Move to an unvisited vault neighbor if possible
        for nb in neighbors:
            if (nb in self.vaults
                    and nb not in self.collected_vaults
                    and nb not in self.dragons):
                if nb in self.suss:
                    self.suss.remove(nb)
                    self.safe.add(nb)
                    return ("destroy", nb)
                else:
                    # We can move here safely
                    self.visited.add(nb)
                    self.harry_loc = nb  # Update Harry's position
                    return ("move", nb)

        # 5) Move to the closest uncollected vault, if exists
        (x, y) = self.harry_loc
        uncollected_vaults = set(x for x in self.vaults if x not in self.collected_vaults)
        if uncollected_vaults:
            closest_uncollected_vault = self.get_closest_vault(uncollected_vaults)
            (r, c) = closest_uncollected_vault
            if r > x:
                direction = (x + 1, y)
                if direction in self.visited:
                    self.harry_loc = direction
                    self.visited.add(direction)
                    return ("move", direction)

            elif r < x:
                direction = (x - 1, y)
                if direction in self.visited:
                    self.harry_loc = direction
                    self.visited.add(direction)
                    return ("move", direction)

            elif c > y:
                direction = (x, y + 1)
                if direction in self.visited:
                    self.harry_loc = direction
                    self.visited.add(direction)
                    return ("move", direction)

            elif c < y:
                direction = (x, y - 1)
                if direction in self.visited:
                    self.harry_loc = direction
                    self.visited.add(direction)
                    return ("move", direction)

        # 6) Explore an unvisited neighbor
        for nb in neighbors:
            if nb in self.safe and nb not in self.visited:
                # We can move here safely
                self.visited.add(nb)
                self.harry_loc = nb
                return ("move", nb)

        # 7) Prefer neighbors in self.suss
        for nb in neighbors:
            if nb in self.suss:
                # Destroy it. We'll mark it safe for next time in _update_knowledge or here
                self.suss.remove(nb)
                self.safe.add(nb)
                return ("destroy", nb)

        # 8) Move to explore the board
        (x, y) = self.harry_loc
        if set(neighbors).issubset(self.visited.union(self.dragons)):
            unexplored_cells = set(x for x in (self.suss.union(self.safe)) if x not in self.visited)
            if unexplored_cells:
                (r, c) = next(iter(unexplored_cells))
                if unexplored_cells:
                    if r > x:
                        direction = (x + 1, y)
                        if direction not in self.dragons:
                            self.harry_loc = direction
                            self.visited.add(direction)
                            return ("move", direction)

                    elif r < x:
                        direction = (x - 1, y)
                        if direction not in self.dragons:
                            self.harry_loc = direction
                            self.visited.add(direction)
                            return ("move", direction)

                    elif c > y:
                        direction = (x, y + 1)
                        if direction not in self.dragons:
                            self.harry_loc = direction
                            self.visited.add(direction)
                            return ("move", direction)

                    elif c < y:
                        direction = (x, y - 1)
                        if direction not in self.dragons:
                            self.harry_loc = direction
                            self.visited.add(direction)
                            return ("move", direction)

        # 9) Move back to where you came from
        for nb in neighbors:
            if nb in self.safe:
                self.visited.add(nb)
                self.harry_loc = nb
                return ("move", nb)

        # 10) If all neighbors are dragons, wait
        if all(nb in self.dragons for nb in neighbors):
            return ("wait",)

    # -------------------------------------------------------------------------
    #                           INTERNAL METHODS
    # -------------------------------------------------------------------------

    def _update_knowledge(self, observations):

        # Process direct observations
        for obs in observations:
            if obs[ 0 ] == "dragon":
                self.dragons.add(obs[ 1 ])  # Mark tile as dragon
                if obs[ 1 ] in self.safe:  # If we had it as safe, remove it
                    self.safe.remove(obs[ 1 ])
            elif obs[ 0 ] == "vault":
                self.vaults.add(obs[ 1 ])

        # Check if we have sulfur in the observations
        sulfur_found = any(obs[ 0 ] == "sulfur" for obs in observations)
        if sulfur_found:
            # All neighbors are suss
            for nb in self._get_neighbors(self.harry_loc):
                if nb not in self.dragons:
                    if nb not in self.safe:
                        self.suss.add(nb)
        else:
            for nb in self._get_neighbors(self.harry_loc):
                if nb not in self.dragons:
                    self.safe.add(nb)
                    if nb in self.suss:
                        self.suss.remove(nb)

    def _get_neighbors(self, loc):
        (r, c) = loc
        rows, cols = self.map_shape
        candidates = [ (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1) ]
        return [
            (nr, nc) for (nr, nc) in candidates
            if 0 <= nr < rows and 0 <= nc < cols
        ]

    def get_closest_vault(self, vaults):
        def manhattan_distance(loc1, loc2):
            return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

        closest_vault = None
        min_distance = float('inf')

        for v in vaults:
            distance = manhattan_distance(self.harry_loc, v)
            if distance < min_distance:
                min_distance = distance
                closest_vault = v

        return closest_vault

# For local testing with checker.py, e.g.:
if __name__ == '__main__':
    from checker import GringottsChecker
    import inputs

    for i, inp in enumerate(inputs.inputs, start=1):
        checker = GringottsChecker(inp)
        result = checker.check_controller()
        print(f"Input {i} => {result}")
