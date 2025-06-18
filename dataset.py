# 2025 - copyright - all rights reserved - clayton thomas baber

import os, tqdm
from torch import tensor, load, save, float32
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from cube import Cube

class RubikDistanceDataModule(LightningDataModule):
    def __init__(self, train_batch, val_batch):
        super().__init__()
        self.train_batch = train_batch
        self.val_batch = val_batch
        
    def train_dataloader(self):
        return DataLoader(
                    RubikDistance(),
                    batch_size = self.train_batch,
                    shuffle = True,
                    num_workers = 0)

    def val_dataloader(self):
        return DataLoader(
                    RubikDistance(),
                    batch_size = self.val_batch,
                    shuffle = False,
                    num_workers = 0)

class RubikDistance(Dataset):
    def __init__(self, data_dir="precomputed_rubiks_data"):
        self.data_dir = data_dir
        self.inputs_path = os.path.join(self.data_dir, "rubiks_inputs.pt")
        self.targets_path = os.path.join(self.data_dir, "rubiks_targets.pt")

        if not os.path.exists(self.inputs_path) or not os.path.exists(self.targets_path):
            os.makedirs(self.data_dir, exist_ok=True) # Create directory if it doesn't exist

            all_inputs = []
            all_targets = []

            cube = Cube()
            npath_idx = 0
            ipath_idx = 0

            contents = set()

            print(f"Generating data samples...")
            for _ in tqdm.tqdm(range(3840 * 20)):
                # Logic to iterate through orbits and generate samples, as in your __getitem__
                if ipath_idx >= len(Cube.orbits[npath_idx]):
                    npath_idx += 1
                    if npath_idx >= len(Cube.orbits):
                        npath_idx = 0
                    ipath_idx = 0
                    cube.reset() # Reset cube when moving to a new 'root' of a path

                # Apply the move and get the state
                current_move = Cube.orbits[npath_idx][ipath_idx]
                cube.act(current_move)

                if cube.getState() not in contents:
                    
                    contents.add(cube.getState())
                    
                    # The input (cube state)
                    input_vector = cube.toOneHot() # This should be your 324-element list/array

                    # The target (distance down the path, current ipath_idx + 1)
                    # Assuming distance is 1-indexed (1 to len(path))
                    # If your model expects 0-indexed distance, adjust accordingly (e.g., ipath_idx)
                    target_distance = ipath_idx + 1 # Distance starts from 1 for the first move

                    all_inputs.append(input_vector)
                    all_targets.append(target_distance)

                ipath_idx += 1

            print(f"Generated {len(all_inputs)} samples.")

            # Convert to PyTorch tensors
            inputs_tensor = tensor(all_inputs, dtype=float32)
            targets_tensor = tensor(all_targets, dtype=float32) # Ensure float32 for MSELoss

            # Save the tensors
            inputs_path = os.path.join(self.data_dir, "rubiks_inputs.pt")
            targets_path = os.path.join(self.data_dir, "rubiks_targets.pt")

            save(inputs_tensor, inputs_path)
            save(targets_tensor, targets_path)

            print(f"Data saved to {inputs_path} and {targets_path}")

        print(f"Loading data from {self.data_dir}...")
        self.inputs = load(self.inputs_path)
        self.targets = load(self.targets_path)
        print(f"Data loaded: {len(self.inputs)} samples.")

        # Ensure inputs are float and targets are float (for MSELoss)
        # This is typically handled during generation and saving, but good to double-check.
        self.inputs = self.inputs.float()
        self.targets = self.targets.float()

        self.size = len(self.inputs) # The dataset size is now fixed by the loaded data


    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
        


    def __len__(self):
        return self.size

if __name__ == "__main__":
    module = RubikDistanceDataModule(1, 1)

    train_sample = iter(module.train_dataloader())
    
    for i, sample in enumerate(train_sample):
        print("train sample:\n", sample)
    
    print("\nval sample:\n", next(iter(module.val_dataloader())))
