# 2023 - copyright - all rights reserved - clayton thomas baber

import torch
from torch import Generator
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
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
            return torch.tensor(next(self.perms)).float(), one_hot(torch.tensor(self.action),18).float()
        except StopIteration:
            self.action = self.path[0]
            self.path = self.path[1:]
            self.perms = RandomColorPermutor(self.gen, [self.start, self.cube.getState(), self.finish], self.colors)
            self.cube.act(self.action)
            return torch.tensor(next(self.perms)).float(), one_hot(torch.tensor(self.action),18).float()

    def __len__(self):
        return self.size

class RandomColorPermutor():
    base = [0]*9 + [1]*9 + [2]*9 + [3]*9 + [4]*9 + [5]*9
    color_hot = ((0, 0, 0),(0, 0, 1),(0, 1, 0),(0, 1, 1),(1, 0, 0),(1, 0, 1),(1, 1, 0),(1, 1, 1))
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
        out = [int(i) for i in list(torch.randperm(8, generator=self.gen, dtype=torch.uint8))][:6]
        out = [[out[i] for i in cube] for cube in self.cubes]
        out = [item for sublist in out for item in sublist]
        out = [RandomColorPermutor.color_hot[i] for i in out]
        out = [item for sublist in out for item in sublist]
        return out

    def __iter__(self):
        return self

if __name__ == "__main__":
    print(next(iter(BrownianAntipodalPaths(1, 0, 1))))
