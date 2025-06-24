# 2025 - copyright - all rights reserved - clayton thomas baber

import os
from tqdm import tqdm
from torch import tensor, load, save, float32
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from cube import Cube
from collections import defaultdict

class RubikDistanceDataModule(LightningDataModule):
    def __init__(self, train_batch, val_batch):
        super().__init__()
        self.train_batch = train_batch
        self.val_batch = val_batch
        
    def train_dataloader(self):
        return DataLoader(
                    RubikDistanceAugmentedData(),
                    batch_size = self.train_batch,
                    shuffle = True,
                    num_workers = 4)

    def val_dataloader(self):
        return DataLoader(
                    RubikDistanceData(),
                    batch_size = self.val_batch,
                    shuffle = False,
                    num_workers = 1)

class RubikDistanceAugmentedData(Dataset):
    def __init__(self, data_dir="precomputed_rubiks_data"):
        self.data_dir = data_dir
        self.inputs_path = os.path.join(self.data_dir, "rubiks_inputs_augmented.pt")
        self.targets_path = os.path.join(self.data_dir, "rubiks_targets_augmented.pt")

        if not os.path.exists(self.inputs_path) or not os.path.exists(self.targets_path):
            os.makedirs(self.data_dir, exist_ok=True)
            
            cube = Cube()
            contents = defaultdict(lambda: 21)
            contents[cube.getState()] = 0

            with tqdm(total=sum(len(sublist) for sublist in Cube.orbits), desc="Generating Data") as pbar:
                for orbit in Cube.orbits:
                    cube.reset()
                    for i, step in enumerate(orbit, start=1):
                        cube.act(step)
                        key = cube.getState()
                        contents[key] = min(contents[key], i)

                        for adj in cube.getAdjacent():
                            key = adj.getState()
                            contents[key] = min(contents[key], i + 1)

                        pbar.update(1)

            inputs = []
            outputs = []
            variants = [
                [1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 1],
                [3, 4, 5, 6, 1, 2],
                [4, 5, 6, 1, 2, 3],
                [5, 6, 1, 2, 3, 4],
                [6, 1, 2, 3, 4, 5]
            ]

            with tqdm(total=len(contents) * len(variants), desc="Augmenting Data") as pbar:
                for color in variants:
                    for k, v in contents.items():
                        worker = Cube()
                        worker.setState(k)
                        
                        inputs.append(worker.toOneHot(color))
                        outputs.append(v)
                        pbar.update(1)

            inputs_tensor = tensor((inputs), dtype=float32)
            targets_tensor = tensor((outputs), dtype=float32)
            
            inputs_path = os.path.join(self.data_dir, "rubiks_inputs_augmented.pt")
            targets_path = os.path.join(self.data_dir, "rubiks_targets_augmented.pt")

            save(inputs_tensor, inputs_path)
            save(targets_tensor, targets_path)

            print(f"Data saved to {inputs_path} and {targets_path}")

        print(f"Loading data from {self.data_dir}...")
        self.inputs = load(self.inputs_path)
        self.targets = load(self.targets_path)
        print(f"Data loaded: {len(self.inputs)} samples.")

        self.inputs = self.inputs.float()
        self.targets = self.targets.float()

        self.size = len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def __len__(self):
        return self.size

class RubikDistanceData(Dataset):
    def __init__(self, data_dir="precomputed_rubiks_data"):
        self.data_dir = data_dir
        self.inputs_path = os.path.join(self.data_dir, "rubiks_inputs.pt")
        self.targets_path = os.path.join(self.data_dir, "rubiks_targets.pt")

        if not os.path.exists(self.inputs_path) or not os.path.exists(self.targets_path):
            os.makedirs(self.data_dir, exist_ok=True)
            
            cube = Cube()
            contents = defaultdict(lambda: 21)
            contents[tuple(cube.toOneHot())] = 0

            with tqdm(total=sum(len(sublist) for sublist in Cube.orbits), desc="Generating Data") as pbar:
                for orbit in Cube.orbits:
                    cube.reset()
                    for i, step in enumerate(orbit, start=1):
                        cube.act(step)
                        key = tuple(cube.toOneHot())
                        contents[key] = min(contents[key], i)

                        adjs = cube.getAdjacent()
                        for adj in adjs:
                            key = tuple(adj.toOneHot())
                            contents[key] = min(contents[key], i + 1)

                        pbar.update(1)

            inputs_tensor = tensor(list(contents.keys()), dtype=float32)
            targets_tensor = tensor(list(contents.values()), dtype=float32)
            
            inputs_path = os.path.join(self.data_dir, "rubiks_inputs.pt")
            targets_path = os.path.join(self.data_dir, "rubiks_targets.pt")

            save(inputs_tensor, inputs_path)
            save(targets_tensor, targets_path)

            print(f"Data saved to {inputs_path} and {targets_path}")

        print(f"Loading data from {self.data_dir}...")
        self.inputs = load(self.inputs_path)
        self.targets = load(self.targets_path)
        print(f"Data loaded: {len(self.inputs)} samples.")

        self.inputs = self.inputs.float()
        self.targets = self.targets.float()

        self.size = len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def __len__(self):
        return self.size

if __name__ == "__main__":
    module = RubikDistanceDataModule(1, 1)

    print("train sample:\n", next(iter(module.train_dataloader())))
    print("\nval sample:\n", next(iter(module.val_dataloader())))
