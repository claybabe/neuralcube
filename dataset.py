# 2026 - copyright - all rights reserved - clayton thomas baber

import os
from tqdm import tqdm
from torch import tensor, load, save, float32, from_numpy, log
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning import LightningDataModule
from cube import Cube
import numpy as np
from collections import deque

# --- CONFIGURATION ---
DATA_DIR = "precomputed_rubiks_data"
INPUTS_FILE = os.path.join(DATA_DIR, "inputs.bin")
TARGETS_FILE = os.path.join(DATA_DIR, "targets.bin")
META_FILE = os.path.join(DATA_DIR, "metadata.pt")

class RubikDistanceDataModule(LightningDataModule):
  def __init__(self, data_dir="precomputed_rubiks_data", train_batch_size=1024, val_batch_size=1024, train_split=0.9, num_workers=4):
    super().__init__()
    self.data_dir = data_dir
    self.train_batch_size = train_batch_size
    self.val_batch_size = val_batch_size
    self.train_split = train_split
    self.num_workers = num_workers

  def setup(self, stage=None):
    entire_dataset = RubikMmapDataset()
    meta = load(META_FILE)
    counts = meta['counts']
    
    # Identify Anchor distances that MUST be in training
    anchors = {0, 1, 2, 3}
    
    train_indices = []
    val_indices = []
    current_offset = 0

    # Iterate through the sorted buckets
    for d in range(21):
      count = counts.get(d, 0)
      if count == 0: continue
      
      # Get the range of indices for this specific distance
      d_indices = np.arange(current_offset, current_offset + count)
      # Manifold Rule: Stratified 90/10 split
      np.random.shuffle(d_indices)
      split = int(count * self.train_split)
        
      if d in anchors:
        # Anchor Rule: 100% to training
        train_indices.append(d_indices)
      else:
        train_indices.append(d_indices[:split])
        val_indices.append(d_indices[split:])
        
      current_offset += count

    self.train_indices = np.concatenate(train_indices)
    self.val_indices = np.concatenate(val_indices)

    # Calculate Log-Smoothed weights for the CrossEntropy
    # Using the counts we saved in metadata
    c_list = tensor([counts.get(i, 0) for i in range(21)], dtype=float32)
    self.class_weights = log((c_list.max() / (c_list + 1e-6)) + 1)

    self.train_ds = Subset(entire_dataset, self.train_indices)
    self.val_ds = Subset(entire_dataset, self.val_indices)

  def train_dataloader(self):
    return DataLoader(
      self.train_ds,
      batch_size=self.train_batch_size,
      shuffle=True, # Important!
      num_workers=self.num_workers,
      pin_memory=True,
      persistent_workers=True # Keeps workers alive between epochs
    )

  def val_dataloader(self):
    return DataLoader(
      self.val_ds,
      batch_size=self.val_batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      pin_memory=True,
      persistent_workers=True
    )

class RubikManager:
  """Handles RAM-efficient generation and disk-based loading."""
  
  @staticmethod
  def generate_dataset(paths, deep_layers=1):
    """
    Generates data using a byte-packed dictionary.
    deep_layers: How many extra steps to explore from every point in the path.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. RAM-Efficient Storage
    # key: bytes(54 stickers), value: uint8 (distance)
    contents = {}
    
    # Seed with solved state
    solved_cube = Cube()
    contents[bytes(solved_cube.state)] = 0

    # Phase 1: path Traversal
    for path in tqdm(paths, desc="Processing Paths"):
      cube = Cube()
      cube.reset()
      for i, action in enumerate(path[:-1], start=1):
        cube.act(action)
        
        # Current state on path
        state_bytes = bytes(cube.state)
        if state_bytes not in contents or i < contents[state_bytes]:
          contents[state_bytes] = i
        
        # Phase 2: "Thickening" (BFS expansion from path)
        # We use a simple frontier to go 'deep_layers' steps away
        frontier = deque([(cube.state, i)])
        for _ in range(deep_layers):
          for _ in range(len(frontier)):
            curr_state, curr_dist = frontier.popleft()
            
            # Use a temporary cube to find adjacents
            temp_cube = Cube()
            temp_cube.setState(curr_state)
            
            for adj in temp_cube.getAdjacent():
              adj_bytes = bytes(adj.state)
              new_dist = curr_dist + 1
              if adj_bytes not in contents or new_dist < contents[adj_bytes]:
                contents[adj_bytes] = new_dist
                # If you wanted to go even deeper, you'd add to frontier here
                if new_dist < 20:
                  frontier.append((adj.state, new_dist))

    # 2. Calculate Statistics & Sort by Distance
    num_samples = len(contents)
    distance_counts = {}
    
    # Sort the items so distance 0 comes first, then 1, etc.
    # This creates the "Physical Buckets" on disk
    sorted_items = sorted(contents.items(), key=lambda x: x[1])
    
    # 2. Write to Disk & Count
    inputs_mmap = np.memmap(INPUTS_FILE, dtype='uint8', mode='w+', shape=(num_samples, 54))
    targets_mmap = np.memmap(TARGETS_FILE, dtype='uint8', mode='w+', shape=(num_samples,))

    for idx, (state_bytes, dist) in enumerate(sorted_items):
      inputs_mmap[idx] = np.frombuffer(state_bytes, dtype='uint8')
      targets_mmap[idx] = dist
      distance_counts[dist] = distance_counts.get(dist, 0) + 1
    
    inputs_mmap.flush()
    targets_mmap.flush()
    
    # 3. Save Metadata with Statistics
    # We save the 'counts' so we can calculate weights without re-scanning
    save({
        'num_samples': num_samples,
        'counts': distance_counts
    }, META_FILE)
    
    print(f"Done! Distribution: {distance_counts}")

    

class RubikMmapDataset(Dataset):
  """Zero-RAM Dataset that reads directly from disk."""
  def __init__(self):
    if not os.path.exists(META_FILE):
      raise FileNotFoundError("Metadata not found. Run generation first.")
      
    meta = load(META_FILE)
    self.num_samples = meta['num_samples']
    
    # mode='r' means we don't load into RAM; we map the file to virtual memory
    self.inputs = np.memmap(INPUTS_FILE, dtype='uint8', mode='r', shape=(self.num_samples, 54))
    self.targets = np.memmap(TARGETS_FILE, dtype='uint8', mode='r', shape=(self.num_samples,))

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    # 1. Get raw sticker indices (0-53)
    state_ints = self.inputs[idx]
    target = int(self.targets[idx])

    # 2. Map stickers to colors (0-5)
    # Based on your Cube class: stickers 0-8 are color 0, 9-17 are color 1, etc.
    # floor division by 9 gives us the face/color index
    color_indices = state_ints // 9 

    # 3. Convert to One-Hot
    one_hot = np.zeros((54, 6), dtype=np.float32)
    one_hot[np.arange(54), color_indices] = 1.0
    
    return from_numpy(one_hot.flatten()), tensor(target, dtype=float32)

class PathDatasetProcessor:
    def __init__(
        self, 
        filepath: str, 
        num_select: int, 
        max_shared_prefix: int = 3, 
        prefix_decay: float = 0.85,
        start_idx: int | str = "random",
        random_seed: int | None = None
    ):
        self.paths = self._load_dataset(filepath)
        self.paths = self._filter_trie_prefixes(self.paths, max_shared_prefix)
        
        # Pass seed config to FPS
        self.paths = self._run_farthest_point_sampling(
            self.paths, 
            num_select, 
            prefix_decay, 
            start_idx=start_idx, 
            random_seed=random_seed
        )
        
        self.paths = self._generate_cyclic_shifts(self.paths)
        self.paths = self._expand_with_reversed_antiactions(self.paths)
        
        rotation_table = self._build_action_rotation_table()
        self.paths = self._expand_paths_with_rotations(self.paths, rotation_table)
        self.paths = np.unique(self.paths, axis=0)

    def _load_dataset(self, filepath: str) -> np.ndarray:
        parsed_paths = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                actions = [line[i:i+2] for i in range(0, len(line), 2)]
                path = [Cube.notation[action] for action in actions]
                parsed_paths.append(path)
        return np.array(parsed_paths, dtype=np.uint8)

    def _filter_trie_prefixes(self, paths: np.ndarray, max_shared_prefix: int) -> np.ndarray:
        trie = {}
        unique_indices = []
        
        for idx, path in enumerate(paths):
            current_node = trie
            is_redundant = False
            
            for depth, move_id in enumerate(path[:max_shared_prefix]):
                move_id = int(move_id)
                if move_id not in current_node:
                    current_node[move_id] = {}
                current_node = current_node[move_id]
                
                current_node['_count'] = current_node.get('_count', 0) + 1
                if depth == (max_shared_prefix - 1) and current_node['_count'] > 1:
                    is_redundant = True
            
            if not is_redundant:
                unique_indices.append(idx)
                
        return paths[unique_indices]

    def _run_farthest_point_sampling(
        self, 
        candidates: np.ndarray, 
        num_select: int, 
        prefix_decay: float = 0.85,
        start_idx: int | str = "random",
        random_seed: int | None = None
    ) -> np.ndarray:
        num_candidates, path_length = candidates.shape
        num_select = min(num_select, num_candidates)
        
        # 1. Resolve starting index
        if start_idx == "random":
            rng = np.random.default_rng(random_seed)
            first_idx = rng.integers(0, num_candidates)
        else:
            first_idx = int(start_idx) % num_candidates

        weights = (prefix_decay ** np.arange(path_length, dtype=np.float32))
        min_distances = np.full(num_candidates, fill_value=np.inf, dtype=np.float32)
        
        selected_indices = [first_idx]
        selected_matrix = np.zeros((num_select, path_length), dtype=np.uint8)
        selected_matrix[0] = candidates[first_idx]
        
        # Initialize distances relative to the chosen first_idx
        mismatches = (candidates != candidates[first_idx])
        min_distances = np.sum(mismatches * weights, axis=1)
        
        # 2. Max-Min selection loop
        for i in range(1, num_select):
            # Select the candidate farthest from all currently selected points
            next_idx = np.argmax(min_distances)
            
            selected_indices.append(next_idx)
            selected_matrix[i] = candidates[next_idx]
            
            # Update running minimum distance to the newly added centroid
            mismatches = (candidates != candidates[next_idx])
            distances_to_latest = np.sum(mismatches * weights, axis=1)
            min_distances = np.minimum(min_distances, distances_to_latest)
            
        return candidates[selected_indices]

    def _generate_cyclic_shifts(self, paths: np.ndarray) -> np.ndarray:
        # Fully vectorized cyclic roll across axis 1
        n_paths, path_len = paths.shape
        shifts = [np.roll(paths, -i, axis=1) for i in range(path_len)]
        return np.vstack(shifts)

    def _expand_with_reversed_antiactions(self, paths: np.ndarray) -> np.ndarray:
        num_actions = len(Cube.actions)
        
        # Convert actions list of lists into a 2D matrix: shape (num_actions, 54)
        actions_mat = np.array(Cube.actions, dtype=np.int32)
        
        # Identify the inverse for each action i
        # action i followed by action j should yield the identity permutation range(54)
        # composition: actions_mat[j, actions_mat[i]] == np.arange(54)
        identity = np.arange(actions_mat.shape[1])
        antiaction_map = np.zeros(num_actions, dtype=np.uint8)
        
        for i in range(num_actions):
            for j in range(num_actions):
                if np.array_equal(actions_mat[j, actions_mat[i]], identity):
                    antiaction_map[i] = j
                    break
        
        # Reverse along path axis (axis=1) and map to anti-actions
        reversed_paths = antiaction_map[np.flip(paths, axis=1)]
        
        return np.vstack((paths, reversed_paths))

    def _build_action_rotation_table(self) -> np.ndarray:
        num_rotations = len(Cube.rotations)
        num_actions = len(Cube.actions)
        table = np.zeros((num_rotations, num_actions), dtype=np.uint8)
        
        for r in range(num_rotations):
            for a in range(num_actions):
                c_ref = Cube()
                c_ref.act(a)
                c_ref.rotate(r)
                ref_state = c_ref.getState()
                
                for test_a in range(num_actions):
                    c_test = Cube()
                    c_test.rotate(r)
                    c_test.act(test_a)
                    if c_test.getState() == ref_state:
                        table[r, a] = test_a
                        break
        return table

    def _expand_paths_with_rotations(self, paths: np.ndarray, rotation_table: np.ndarray) -> np.ndarray:
        # NumPy advanced indexing applies all 24 transformation rows simultaneously
        # Input shape: (N, L) -> Output shape: (24, N, L)
        expanded = rotation_table[:, paths]
        # Reshape into a flat batch of paths: (24 * N, L)
        return expanded.reshape(-1, paths.shape[1])

    def get_paths(self) -> np.ndarray:
        return self.paths.tolist()

# --- EXAMPLE USAGE ---
if __name__ == "__main__":

  # 0. SELECT INITIAL SEED PATHS  
  pdp = PathDatasetProcessor("htm4.txt", 4, max_shared_prefix=6)
  paths = pdp.get_paths()

  # 1. GENERATE
  manager = RubikManager()
  manager.generate_dataset(paths, deep_layers=2) # Comment out if already generated

  # 2. INITIALIZE DATA MODULE
  # Note: Use your actual desired batch_size here, e.g., 1024
  module = RubikDistanceDataModule(train_batch_size=6, val_batch_size=1, train_split=0.5)

  # 3. MANUAL SETUP
  # This is the line you were missing! 
  # Lightning normally does this for you, but for manual testing, it's required.
  module.setup() 

  # 4. TEST LOADERS
  train_loader = module.train_dataloader()
  val_loader = module.val_dataloader()

  inputs, targets = next(iter(train_loader))
  print("Train batch inputs shape:", inputs.shape)
  print("Train batch targets shape:", targets.shape)
  
  for pair in zip(inputs, targets):
    print(f"{pair}\n\n")

  del train_loader
  del val_loader
  print("done with test")