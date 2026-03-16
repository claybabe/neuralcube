# 2025 - copyright - all rights reserved - clayton thomas baber

import os
from tqdm import tqdm
from torch import tensor, load, save, float32, from_numpy
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
    # 1. Initialize the full dataset (this just maps the files, doesn't load them)
    entire_dataset = RubikMmapDataset()
    dataset_size = len(entire_dataset)
    
    # 2. Create shuffled indices for the split
    indices = np.arange(dataset_size)
    np.random.seed(42) # For reproducible splits
    np.random.shuffle(indices)
    
    split_point = int(dataset_size * self.train_split)
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    # 3. Create Subsets
    # Subsets are great because they don't copy the data, just the index list
    self.train_ds = Subset(entire_dataset, train_indices)
    self.val_ds = Subset(entire_dataset, val_indices)

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
  def generate_dataset(orbits, deep_layers=1):
    """
    Generates data using a byte-packed dictionary.
    deep_layers: How many extra steps to explore from every point in the orbit.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. RAM-Efficient Storage
    # key: bytes(54 stickers), value: uint8 (distance)
    contents = {}
    
    # Seed with solved state
    solved_cube = Cube()
    contents[bytes(solved_cube.state)] = 0

    # Phase 1: Orbit Traversal
    for orbit in tqdm(orbits, desc="Processing Orbits"):
      cube = Cube()
      cube.reset()
      for i, action in enumerate(orbit, start=1):
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
                frontier.append((adj.state, new_dist))

    # 2. Finalize to Disk (Memory Mapping)
    num_samples = len(contents)
    print(f"Finalizing {num_samples} samples to SSD...")
    
    # Pre-allocate binary files
    inputs_mmap = np.memmap(INPUTS_FILE, dtype='uint8', mode='w+', shape=(num_samples, 54))
    targets_mmap = np.memmap(TARGETS_FILE, dtype='uint8', mode='w+', shape=(num_samples,))

    for idx, (state_bytes, dist) in enumerate(contents.items()):
      inputs_mmap[idx] = np.frombuffer(state_bytes, dtype='uint8')
      targets_mmap[idx] = dist
    
    inputs_mmap.flush()
    targets_mmap.flush()
    
    # Save metadata so we know the size when reloading
    save({'num_samples': num_samples}, META_FILE)
    print(f"Done! Data saved to {DATA_DIR}")

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

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
  # 1. GENERATE
  manager = RubikManager()
  manager.generate_dataset(Cube.orbits, deep_layers=1) # Comment out if already generated

  # 2. INITIALIZE DATA MODULE
  # Note: Use your actual desired batch_size here, e.g., 1024
  module = RubikDistanceDataModule(batch_size=1024)

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
  print("First target in batch:", targets[0].item())

  del train_loader
  del val_loader
  print("done with test")