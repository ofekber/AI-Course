

class GringottsController:
    def __init__(self, map_shape, harry_loc, initial_observations):
        """
        :param map_shape: (rows, cols) map dimensions
        :param harry_loc: (r, c) starting position of Harry
        :param initial_observations: observations from the start tile
        """
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
        """
        Called each turn with the new observations from our current location.
        1) Update knowledge about dragons, traps, vaults, safe squares, suspicious squares.
        2) If we're standing on a new vault => collect.
        3) Decide on a single-step move among neighbors (no wait):
           a) If a neighbor is an unvisited vault => move there
           b) Else if a neighbor is safe & unvisited => move there
           c) Else destroy a suspicious/unknown neighbor so we can potentially move there next turn
        """

        # 1) Update knowledge
        self._update_knowledge(observations)

        # 2) If we are on a new vault => collect
        if self.harry_loc in self.vaults and self.harry_loc not in self.collected_vaults:
            self.collected_vaults.add(self.harry_loc)
            return ("collect",)  # if it's the hallow vault, game ends

        # 3) Attempt a local move
        neighbors = self._get_neighbors(self.harry_loc)

        # 3a) Move to an unvisited vault neighbor if possible
        for nb in neighbors:
            if (nb in self.vaults
                    and nb not in self.collected_vaults
                    and nb not in self.dragons
                    and nb not in self.suss):
                # We can move here safely
                self.visited.add(nb)
                self.harry_loc = nb  # Update Harry's position
                return ("move", nb)

        # 3b) Otherwise, move to a safe, unvisited neighbor
        for nb in neighbors:
            if nb in self.safe and nb not in self.visited:
                # We can move here safely
                self.visited.add(nb)
                self.harry_loc = nb
                return ("move", nb)

        # 4) If no unvisited vault or safe neighbor, then destroy a suspicious/unknown neighbor.
        #    This is how we avoid "wait" and possibly clear a trap.
        #    We'll pick any neighbor that isn't known safe or dragon.
        #    If you store suspicious squares in self.suss, use that to find a candidate.

        # Example approach: prefer neighbors in self.suss
        for nb in neighbors:
            if nb in self.suss:
                # Destroy it. We'll mark it safe for next time in _update_knowledge or here
                self.suss.remove(nb)
                self.safe.add(nb)
                return ("destroy", nb)

        # 3b) Otherwise, move to a safe, unvisited neighbor
        for nb in neighbors:
            if nb in self.safe:
                # We can move here safely
                self.visited.add(nb)
                self.harry_loc = nb
                return ("move", nb)

    # -------------------------------------------------------------------------
    #                           INTERNAL METHODS
    # -------------------------------------------------------------------------

    def _update_knowledge(self, observations):
        """
        Update our knowledge about the environment:
          - If no sulfur => neighbors are safe (unless known dragon).
          - If we see 'dragon' => mark that tile as dragon
          - If we see 'vault' => record
          - We do no advanced 'sulfur => some neighbor is trap' inference here,
            so we won't label squares as traps unless we have direct evidence.
        """

        # Process direct observations
        for obs in observations:
            if obs[0] == "dragon":
                self.dragons.add(obs[1])  # Mark tile as dragon
                if obs[1] in self.safe:  # If we had it as safe, remove it
                    self.safe.remove(obs[1])
            elif obs[0] == "vault":
                self.vaults.add(obs[1])

        # Check if we have sulfur in the observations
        sulfur_found = any(obs[0] == "sulfur" for obs in observations)
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
        """
        Return valid up/down/left/right neighbors in-bounds.
        """
        (r, c) = loc
        rows, cols = self.map_shape
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [
            (nr, nc) for (nr, nc) in candidates
            if 0 <= nr < rows and 0 <= nc < cols
        ]


# For local testing with checker.py, e.g.:
if __name__ == '__main__':
    from checker import GringottsChecker
    import inputs

    for i, inp in enumerate(inputs.inputs, start=1):
        checker = GringottsChecker(inp)
        result = checker.check_controller()
        print(f"Input {i} => {result}")
