# 2025 - copyright - all rights reserved - clayton thomas baber

import os
from tqdm import tqdm
from torch import tensor, load, save, float32
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from cube import Cube
from collections import defaultdict
from itertools import permutations
from random import shuffle, randint

class RubikEncoderDataModule(LightningDataModule):
    def __init__(self, train_batch, val_batch, regenerate=False):
        super().__init__()
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.regenerate = regenerate
        self.train_dataset = RubikEncoderDataset(regenerate=self.regenerate, who="train")
        self.val_dataset = RubikEncoderDataset(regenerate=self.regenerate, who="val", depth=3)

    def train_dataloader(self):
        return DataLoader(
                    self.train_dataset,
                    batch_size = self.train_batch,
                    shuffle = True,
                    num_workers = 4)
    
    def val_dataloader(self):
        return DataLoader(
                    self.val_dataset,
                    batch_size = self.val_batch,
                    shuffle = False,
                    num_workers = 4)

class RubikEncoderDataset(Dataset):
    def __init__(self, 
            data_dir="precomputed_rubiks_data",
            input_name = "rubiks_inputs_augmented.pt",
            targets_name = "rubiks_targets_augmented.pt",
            regenerate=False,
            depth=42,
            who="train"
        ):
        self.data_dir = data_dir
        path = f"rubiks_inputs_encoder_{who}.pt"
        self.inputs_path = os.path.join(self.data_dir, path)
        self.depth = depth
        
        if regenerate or not os.path.exists(self.inputs_path):
            os.makedirs(self.data_dir, exist_ok=True)
            with tqdm(total=(len(Cube.orbits) + 1) * self.depth, desc="Generating Data") as pbar:
                cube = Cube()
                contents = set()
                contents.add(cube.getState())
                for i in range(self.depth):
                    cube.act(randint(0,17))
                    contents.add(cube.getState())
                    pbar.update(1)

                for orbit in Cube.orbits:
                    cube.reset()
                    cube.algo(orbit)
                    contents.add(cube.getState())
                    for i in range(self.depth):
                        cube.act(randint(0,17))
                        contents.add(cube.getState())
                        pbar.update(1)
                    
            inputs = []
            
            with tqdm(total=len(contents) * 720, desc="Augmenting Data") as pbar:
                for state in contents:
                    worker = Cube()
                    worker.setState(state)
                    for color in permutations((1, 2, 3, 4, 5, 6)):
                        inputs.append(worker.toOneHot(color))
                        pbar.update(1)

            inputs_tensor = tensor((inputs), dtype=float32)
            save(inputs_tensor, self.inputs_path)
            print(f"Data saved to {self.inputs_path}")

        print(f"Loading data from {self.data_dir}...")
        self.inputs = load(self.inputs_path)
        
        self.size = len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

    def getAll(self):
        return self.inputs

    def __len__(self):
        return self.size

class RubikDistanceDataModule(LightningDataModule):
    def __init__(self, train_batch, val_batch, regenerate=False):
        super().__init__()
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.regenerate = regenerate
        self.train_dataset = RubikDistanceAugmentedDataset(regenerate=self.regenerate)
        self.val_dataset = RubikDistanceDataset(regenerate=self.regenerate)
        
    def train_dataloader(self):
        return DataLoader(
                    self.train_dataset,
                    batch_size = self.train_batch,
                    shuffle = True,
                    num_workers = 4)

    def val_dataloader(self):
        return DataLoader(
                    self.val_dataset,
                    batch_size = self.val_batch,
                    shuffle = False,
                    num_workers = 4)

class RandomPermutationSubset():
    def __init__(self, base=(1, 2, 3, 4, 5, 6), size=1):
        self.base = base
        self.size = size
        permu = permutations(base)
        next(permu) #skip the first, we add it after choosing shuffeled size - 1 perms
        self.perms = [*permu]
    def __next__(self):
        shuffle(self.perms)
        return [self.base, *self.perms[:self.size - 1]]
    def __len__(self):
        return self.size

class RubikDistanceAugmentedDataset(Dataset):
    def __init__(self,
            data_dir="precomputed_rubiks_data",
            input_name = "rubiks_inputs_augmented.pt",
            targets_name = "rubiks_targets_augmented.pt",
            regenerate=False
        ):
        self.data_dir = data_dir
        self.input_name = input_name
        self.targets_name = targets_name
        self.inputs_path = os.path.join(self.data_dir, self.input_name)
        self.targets_path = os.path.join(self.data_dir, self.targets_name)

        if regenerate or not os.path.exists(self.inputs_path) or not os.path.exists(self.targets_path):
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
            variants = RandomPermutationSubset((1, 2, 3, 4, 5, 6), 1)

            with tqdm(total=len(contents) * len(variants), desc="Augmenting Data") as pbar:
                for k, v in contents.items():
                    worker = Cube()
                    for color in next(variants):
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

class RubikDistanceDataset(Dataset):
    def __init__(self, data_dir="precomputed_rubiks_data", regenerate=False):
        self.data_dir = data_dir
        self.inputs_path = os.path.join(self.data_dir, "rubiks_inputs.pt")
        self.targets_path = os.path.join(self.data_dir, "rubiks_targets.pt")

        if regenerate or not os.path.exists(self.inputs_path) or not os.path.exists(self.targets_path):
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
