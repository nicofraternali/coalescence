from genart_killer import puzzle as p

pz = p.load("puzzles/example_001.yaml")
print(f"Loaded: {pz.puzzle_id}")
print(f"  Cages: {len(pz.cages)}")
print(f"  Cage sums total: {sum(c.sum for c in pz.cages)}")

state = p.initial_state(pz)
print(f"  Cage 1 (sum 14, 2 cells) valid combinations: "
      f"{sorted([sorted(c) for c in state.cage_combinations[1]])}")