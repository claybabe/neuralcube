# 2023 - copyright - all rights reserved - clayton thomas baber

import torch
from torch import Generator
from torch.utils.data import Dataset
from cube import Cube

class BrownianAntipodalPaths(Dataset):
    def __init__(self, size, wander, colors, seed=272497620):
        self.size = size
        self.wander = wander
        self.colors = colors
        self.gen = Generator()
        self.gen.manual_seed(seed)
        self.cube = Cube()
        self.perms  = iter([])
        self.start  = None
        self.finish = None
        self.path   = None
        self.action = None

    def __getitem__(self, idx):
        if not self.path:
            self.cube.algo([int(i) for i in list(torch.randperm(len(Cube.actions), generator=self.gen, dtype=torch.uint8))][:self.wander])
            self.path = Cube.orbits[int(torch.randint(len(Cube.orbits), (1,1), generator=self.gen, dtype=torch.int16))]
            self.start = self.cube.getState()
            self.cube.algo(self.path)
            self.finish = self.cube.getState()
            self.cube.setState(self.start)

        try:
            return torch.tensor([item for sublist in next(self.perms) for item in sublist], dtype=torch.float), torch.tensor(self.action, dtype=torch.float)
        except StopIteration:
            self.action = self.path[0]
            self.path = self.path[1:]
            self.perms = RandomColorPermutor(self.gen, [self.start, self.cube.getState(), self.finish], self.colors)
            self.cube.act(self.action)
            return torch.tensor([item for sublist in next(self.perms) for item in sublist], dtype=torch.float), torch.tensor(self.action, dtype=torch.float)

    def __len__(self):
        return self.size

class RandomColorPermutor():
    base = [0]*9 + [1]*9 + [2]*9 + [3]*9 + [4]*9 + [5]*9
    def __init__(self, gen, cubes, limit):
        self.gen = gen
        self.limit = limit
        self.cubes = cubes
        for i,v in enumerate(self.cubes):
            self.cubes[i] = [RandomColorPermutor.base[i] for i in v]

    def __next__(self):
        if self.limit < 1:
            raise StopIteration
        self.limit -= 1
        perm = [int(i) for i in list(torch.randperm(8, generator=self.gen, dtype=torch.uint8))][:6]
        return [[perm[i] for i in cube] for cube in self.cubes]

    def __iter__(self):
        return self

if __name__ == "__main__":
    print(next(iter(BrownianAntipodalPaths(1, 0, 1))))
