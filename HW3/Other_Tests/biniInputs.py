inputs = [{
    "optimal": True,
    "turns_to_go": 20,  # => (20 + 1) = 21
    "map": [
        # 3 rows, 4 cols
        # We'll mark some cells 'I' (impassable) so that there are exactly 10 passable.
        # Row0: P,P,I,P -> passable=3
        # Row1: P,P,I,P -> passable=3
        # Row2: I,P,P,P -> passable=3 + 1? Actually 3 passable on row2: (2,1),(2,2),(2,3)
        # But we want exactly 10 total. Let's adjust carefully:
        #   Actually let's make row2: P,P,P,P => 4 passable, so total=3+3+4=10
        ["P","P","I","P"],  # passable=3
        ["P","P","I","P"],  # passable=3  => total so far=6
        ["P","P","P","P"],  # passable=4  => total=10
    ],
    # 2 wizards => factor (#passable_cells^2) = 10^2=100
    "wizards": {
        "Wizard_0": {"location": (2,2)},
        "Wizard_1": {"location": (1,3)},
        "Wizard_2": {"location": (2,0)},
        "Wizard_3": {"location": (0,1)}
    },
    # 1 horcrux => factor of #possible_locs=1
    "horcrux": {
        "Horcrux_0": {
            "location": (1,3),
            "possible_locations": [(1,3)],  # exactly 1 possible location
            "prob_change_location": 0.0     # doesn't matter
        }
    },
    # 2 Death Eaters, each path length=69 => product=4761
    "death_eaters": {
        "Death_Eater_0": {
            "index": 0,
            # We'll just store a 69-long path of repeated coords, e.g. (0,0) repeated
            "path": [(0,0)]
        },
        "Death_Eater_1": {
            "index": 0,
            # Another 69-long path, e.g. (2,3) repeated
            "path": [(2,3)]
        }
    },
}
]
