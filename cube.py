# 2026 - copyright - all rights reserved - clayton thomas baber

import zipfile
import io
from tqdm import tqdm
from collections import Counter

class Cube():
  solved = tuple(range(54))
  actions = (
  (42,1,2,43,4,5,44,7,8,0,10,11,3,13,14,6,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,9,12,15,36,37,38,39,40,41,33,34,35,51,48,45,52,49,46,53,50,47),
  (33,1,2,34,4,5,35,7,8,42,10,11,43,13,14,44,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,0,3,6,36,37,38,39,40,41,9,12,15,53,52,51,50,49,48,47,46,45),
  (9,1,2,12,4,5,15,7,8,33,10,11,34,13,14,35,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,42,43,44,36,37,38,39,40,41,0,3,6,47,50,53,46,49,52,45,48,51),
  
  (6,3,0,7,4,1,8,5,2,18,19,20,12,13,14,15,16,17,38,41,44,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,47,39,40,50,42,43,53,45,46,9,48,49,10,51,52,11),
  (8,7,6,5,4,3,2,1,0,38,41,44,12,13,14,15,16,17,47,50,53,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,9,39,40,10,42,43,11,45,46,18,48,49,19,51,52,20),
  (2,5,8,1,4,7,0,3,6,47,50,53,12,13,14,15,16,17,9,10,11,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,18,39,40,19,42,43,20,45,46,38,48,49,41,51,52,44),
  
  (0,1,2,3,4,5,51,52,53,15,12,9,16,13,10,17,14,11,6,19,20,7,22,23,8,25,26,18,28,29,21,31,32,24,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,27,30,33),
  (0,1,2,3,4,5,27,30,33,17,16,15,14,13,12,11,10,9,51,19,20,52,22,23,53,25,26,6,28,29,7,31,32,8,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,18,21,24),
  (0,1,2,3,4,5,18,21,24,11,14,17,10,13,16,9,12,15,27,19,20,30,22,23,33,25,26,51,28,29,52,31,32,53,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,6,7,8),
  
  
  (0,1,36,3,4,37,6,7,38,9,10,2,12,13,5,15,16,8,20,23,26,19,22,25,18,21,24,11,14,17,30,31,32,33,34,35,27,28,29,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53),
  (0,1,27,3,4,28,6,7,29,9,10,36,12,13,37,15,16,38,26,25,24,23,22,21,20,19,18,2,5,8,30,31,32,33,34,35,11,14,17,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53),
  (0,1,11,3,4,14,6,7,17,9,10,27,12,13,28,15,16,29,24,21,18,25,22,19,26,23,20,36,37,38,30,31,32,33,34,35,2,5,8,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53),
  
  (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,24,25,26,18,19,20,21,22,23,36,39,42,29,32,35,28,31,34,27,30,33,45,37,38,48,40,41,51,43,44,15,46,47,16,49,50,17,52,53),
  (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,36,39,42,18,19,20,21,22,23,45,48,51,35,34,33,32,31,30,29,28,27,15,37,38,16,40,41,17,43,44,24,46,47,25,49,50,26,52,53),
  (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,45,48,51,18,19,20,21,22,23,15,16,17,33,30,27,34,31,28,35,32,29,24,37,38,25,40,41,26,43,44,36,46,47,39,49,50,42,52,53),

  (45,46,47,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,0,21,22,1,24,25,2,27,28,20,30,31,23,33,34,26,38,41,44,37,40,43,36,39,42,29,32,35,48,49,50,51,52,53),
  (29,32,35,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,45,21,22,46,24,25,47,27,28,0,30,31,1,33,34,2,44,43,42,41,40,39,38,37,36,20,23,26,48,49,50,51,52,53),
  (20,23,26,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,29,21,22,32,24,25,35,27,28,45,30,31,46,33,34,47,42,39,36,43,40,37,44,41,38,0,1,2,48,49,50,51,52,53))

  rotations = (
  (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53),

  (20,23,26,19,22,25,18,21,24,11,14,17,10,13,16,9,12,15,27,28,29,30,31,32,33,34,35,51,48,45,52,49,46,53,50,47,42,39,36,43,40,37,44,41,38,0,1,2,3,4,5,6,7,8),
  (29,32,35,28,31,34,27,30,33,17,16,15,14,13,12,11,10,9,51,48,45,52,49,46,53,50,47,6,3,0,7,4,1,8,5,2,44,43,42,41,40,39,38,37,36,20,23,26,19,22,25,18,21,24),
  (45,46,47,48,49,50,51,52,53,15,12,9,16,13,10,17,14,11,6,3,0,7,4,1,8,5,2,18,19,20,21,22,23,24,25,26,38,41,44,37,40,43,36,39,42,29,32,35,28,31,34,27,30,33),

  (9,10,11,12,13,14,15,16,17,33,30,27,34,31,28,35,32,29,24,21,18,25,22,19,26,23,20,36,37,38,39,40,41,42,43,44,2,5,8,1,4,7,0,3,6,47,50,53,46,49,52,45,48,51),
  (33,30,27,34,31,28,35,32,29,42,39,36,43,40,37,44,41,38,26,25,24,23,22,21,20,19,18,2,5,8,1,4,7,0,3,6,11,14,17,10,13,16,9,12,15,53,52,51,50,49,48,47,46,45),
  (42,39,36,43,40,37,44,41,38,0,1,2,3,4,5,6,7,8,20,23,26,19,22,25,18,21,24,11,14,17,10,13,16,9,12,15,27,28,29,30,31,32,33,34,35,51,48,45,52,49,46,53,50,47),

  (2,5,8,1,4,7,0,3,6,47,50,53,46,49,52,45,48,51,9,10,11,12,13,14,15,16,17,33,30,27,34,31,28,35,32,29,24,21,18,25,22,19,26,23,20,36,37,38,39,40,41,42,43,44),
  (8,7,6,5,4,3,2,1,0,38,41,44,37,40,43,36,39,42,47,50,53,46,49,52,45,48,51,35,34,33,32,31,30,29,28,27,15,12,9,16,13,10,17,14,11,24,21,18,25,22,19,26,23,20),
  (6,3,0,7,4,1,8,5,2,18,19,20,21,22,23,24,25,26,38,41,44,37,40,43,36,39,42,29,32,35,28,31,34,27,30,33,45,46,47,48,49,50,51,52,53,15,12,9,16,13,10,17,14,11),

  (11,14,17,10,13,16,9,12,15,53,52,51,50,49,48,47,46,45,33,30,27,34,31,28,35,32,29,42,39,36,43,40,37,44,41,38,26,25,24,23,22,21,20,19,18,2,5,8,1,4,7,0,3,6),
  (18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17),
  (17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18),
  (27,28,29,30,31,32,33,34,35,51,48,45,52,49,46,53,50,47,42,39,36,43,40,37,44,41,38,0,1,2,3,4,5,6,7,8,20,23,26,19,22,25,18,21,24,11,14,17,10,13,16,9,12,15),
  (53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0),
  (38,41,44,37,40,43,36,39,42,29,32,35,28,31,34,27,30,33,45,46,47,48,49,50,51,52,53,15,12,9,16,13,10,17,14,11,6,3,0,7,4,1,8,5,2,18,19,20,21,22,23,24,25,26),
  (24,21,18,25,22,19,26,23,20,36,37,38,39,40,41,42,43,44,2,5,8,1,4,7,0,3,6,47,50,53,46,49,52,45,48,51,9,10,11,12,13,14,15,16,17,33,30,27,34,31,28,35,32,29),
  (15,12,9,16,13,10,17,14,11,24,21,18,25,22,19,26,23,20,8,7,6,5,4,3,2,1,0,38,41,44,37,40,43,36,39,42,47,50,53,46,49,52,45,48,51,35,34,33,32,31,30,29,28,27),
  (51,48,45,52,49,46,53,50,47,6,3,0,7,4,1,8,5,2,44,43,42,41,40,39,38,37,36,20,23,26,19,22,25,18,21,24,29,32,35,28,31,34,27,30,33,17,16,15,14,13,12,11,10,9),
  (36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35),
  (44,43,42,41,40,39,38,37,36,20,23,26,19,22,25,18,21,24,29,32,35,28,31,34,27,30,33,17,16,15,14,13,12,11,10,9,51,48,45,52,49,46,53,50,47,6,3,0,7,4,1,8,5,2),
  (47,50,53,46,49,52,45,48,51,35,34,33,32,31,30,29,28,27,15,12,9,16,13,10,17,14,11,24,21,18,25,22,19,26,23,20,8,7,6,5,4,3,2,1,0,38,41,44,37,40,43,36,39,42),
  (26,25,24,23,22,21,20,19,18,2,5,8,1,4,7,0,3,6,11,14,17,10,13,16,9,12,15,53,52,51,50,49,48,47,46,45,33,30,27,34,31,28,35,32,29,42,39,36,43,40,37,44,41,38),
  (35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36))

  notation = {
    "L1" : 0,
    "L2" : 1,
    "L3" : 2,

    "U1" : 3,
    "U2" : 4,
    "U3" : 5,

    "F1" : 6,
    "F2" : 7,
    "F3" : 8, 

    "R3" : 9,
    "R2" : 10,
    "R1" : 11,

    "D3" : 12,
    "D2" : 13,
    "D1" : 14,

    "B3" : 15,
    "B2" : 16,
    "B1" : 17
  }

  def __init__(self, cube=None):
    self.state = Cube.solved if cube is None else cube.state
    self.target = Cube.solved if cube is None else cube.target

  def reset(self):
    self.state = Cube.solved
    self.target = Cube.solved

  def act(self, action):
    self.state = [self.state[i] for i in Cube.actions[action]]

  def rotate(self, rotation):
    self.target = [self.target[i] for i in Cube.rotations[rotation]] 
    self.state = [self.state[i] for i in Cube.rotations[rotation]]

  def algo(self, actions):
    for action in actions:
      self.act(action)

  def getState(self):
    return tuple(self.state)

  def setState(self, state):
    self.state = state

  def isSolved(self):
    return tuple(self.state) == tuple(self.target)

  def toColor(self):
    color_map = tuple([0]*9 + [1]*9 + [2]*9 + [3]*9 + [4]*9 + [5]*9)
    return [color_map[i] for i in self.state]

  def toBipolarHot(self):
    subject = self.toColor()
    bipolar_map = (
        (1, -1, 0),   # 0: Up (White)
        (1, 0, -1),   # 1: Front (Red)
        (0, -1, 1),   # 2: Right (Blue)
        (-1, 1, 0),   # 3: Bottom (Yellow)
        (-1, 0, 1),   # 4: Back (Orange)
        (0, 1, -1),   # 5: Left (Green)
    )
    out = []
    for sticker in subject:
      out += bipolar_map[sticker]
    return out

  def toColorHot(self, L=1):
    # Fixed color tuples matching: White, Red, Blue, Yellow, Orange, Green
    # Scaled by integer L
    color_hot = (
        (L, L, L),        # 0: Up (White)
        (L, 0, 0),        # 1: Front (Red)
        (0, 0, L),        # 2: Right (Blue)
        (L, L, 0),        # 3: Bottom (Yellow)
        (L, int(0.5*L), 0), # 4: Back (Orange)
        (0, L, 0)         # 5: Left (Green)
    )
    subject = self.toColor()
    state = [color_hot[i] for i in subject]
    return state

  def toOneHot(self):
    subject = self.toColor()
    out = []
    for sticker in subject:
      out_hot = [0] * 6
      out_hot[sticker] = 1
      out += out_hot
    return out

  def getProbe(self):
    sprouts = []
    for i in range(18):
      seedling = Cube(self)
      seedling.act(i)
      sprouts.append(seedling.toOneHot())
    return sprouts

  def getAdjacent(self):
    adjs = []
    for i in range(18):
      adj = Cube(self)
      adj.act(i)
      adjs.append(adj)
    return adjs

  def __repr__(self):
    return repr(self.state)

if __name__ == "__main__":
  
  # 1. Notation Cycle Tests
  print("--- Testing Notations ---")
  all_notation_states = set()
  
  for key, action in Cube.notation.items():
    cube = Cube()
    cycle_states = []
    
    # First application
    cube.act(action)
    assert not cube.isSolved(), f"Action {key} did not alter solved state"
    cycle_states.append(cube.getState())
    
    # Complete the first cycle
    while not cube.isSolved():
      cube.act(action)
      cycle_states.append(cube.getState())
      
    cycle_length = len(cycle_states)
    assert len(set(cycle_states)) == cycle_length, f"Duplicate states within single cycle of {key}"
    
    # Apply cycle 2 more times to verify self-consistency
    for _ in range(2 * cycle_length):
      cube.act(action)
    assert cube.isSolved(), f"Cycling {key} 2 more times failed to return to solved state"
    
    # Check uniqueness across all notation cycles
    for state in cycle_states:
      all_notation_states.add(state)
      
    print(f"Notation {key:2s} | Cycle Length: {cycle_length} | PASSED")
    
  print(f"All notation tests passed. Unique states across notation cycles: {len(all_notation_states)}\n")


  # 2. Rotation Cycle Tests
  print("--- Testing Rotations ---")
  all_rotation_states = set()
  
  for idx, rotation in enumerate(Cube.rotations):
    cube = Cube()
    initial_state = cube.getState()
    
    # Verify rotation preserves solved state while altering internal representation
    cube.rotate(idx)
    assert cube.isSolved(), f"Rotation {idx} broke isSolved() state"
    
    if idx == 0:
      # Index 0 is the identity rotation
      assert cube.getState() == initial_state, "Identity rotation altered state"
      cycle_length = 1
    else:
      # Non-identity rotations must produce a distinct state representation
      assert cube.getState() != initial_state, f"Rotation {idx} failed to change state from fresh cube"
      
      cycle_states = [cube.getState()]
      # Perform full cycle back to initial state representation (range(54))
      while cube.getState() != initial_state:
        cube.rotate(idx)
        assert cube.isSolved(), f"Rotation {idx} broke isSolved() during cycling"
        cycle_states.append(cube.getState())
        
      cycle_length = len(cycle_states)
      assert len(set(cycle_states)) == cycle_length, f"Duplicate state representations in cycle for rotation {idx}"
      
      # Repeat cycle 2 more times to verify consistency
      for _ in range(2 * cycle_length):
        cube.rotate(idx)
        assert cube.isSolved(), f"Rotation {idx} broke isSolved() during repeated cycling"
      assert cube.getState() == initial_state, f"Rotation {idx} failed to return to initial state representation"

      for state in cycle_states:
        all_rotation_states.add(state)
      
    print(f"Rotation {idx:2d} | Cycle Length: {cycle_length} | PASSED")

  print(f"All rotation tests passed. Unique state representations across non-identity rotations: {len(all_rotation_states)}\n")


  # 3. Path Cycle Tests via htm4.zip
  print("--- Testing Paths in htm4.zip ---")
  path_cycle_counts = Counter()
  action_distribution = Counter()

  with zipfile.ZipFile("assets/htm4.zip", "r") as zf:
    namelist = zf.namelist()
    txt_files = [f for f in namelist if f.endswith('.txt') and not f.startswith('__MACOSX')]
    target_filename = txt_files[0] if txt_files else namelist[0]

    with zf.open(target_filename, 'r') as f:
      text_stream = io.TextIOWrapper(f, encoding='utf-8')
      for line in tqdm(text_stream, desc=f"Testing paths in '{target_filename}'", unit="lines"):
        line = line.strip()
        if not line:
          continue

        # Parse two-character notation tokens into action indices
        actions_str = [line[i:i+2] for i in range(0, len(line), 2)]
        path = [Cube.notation[action] for action in actions_str]

        # Track action distribution
        for action in path:
          action_distribution[action] += 1

        # Determine path cycle length
        cube = Cube()
        cube.algo(path)

        cycle_len = 1
        while not cube.isSolved():
          cube.algo(path)
          cycle_len += 1

        path_cycle_counts[cycle_len] += 1

  print("Path Cycle Length Distribution:")
  for length, count in sorted(path_cycle_counts.items()):
    print(f"  Length {length:3d}: {count} paths")

  print("\nAction Distribution:")
  for action, count in sorted(action_distribution.items()):
    s = "  " if action < 10 else " "
    print(f"{s}{action}: {count}")

  print("\nAll tests complete successfully.")