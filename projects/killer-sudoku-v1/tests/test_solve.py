from genart_killer import puzzle as p
from genart_killer import solver as s

pz = p.load("puzzles/001_hard.yaml")
result = s.solve(pz)

print(f"Solved: {result.solved}")
print(f"Trace events: {len(result.trace.events)}")
print(f"Stuck cells: {len(result.stuck_cells)}")

result.trace.save(f"traces/{pz.puzzle_id}.json")
print(f"Trace saved to traces/{pz.puzzle_id}.json")