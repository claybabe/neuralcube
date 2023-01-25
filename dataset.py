# 2023 - copyright - all rights reserved - clayton thomas baber

from torch import Generator, tensor, randperm, randint, uint8, int16
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch.nn.functional import one_hot
from cube import Cube

class BrownianAntipodalDataModule(LightningDataModule):
    def __init__(self, train_size, train_batch, train_wander, train_colors, val_size, val_batch, val_wander, val_colors, train_seed=133742247331, val_seed=272497620):
        super().__init__()
        self.train_size, self.train_batch, self.train_wander, self.train_colors = train_size, train_batch, train_wander, train_colors
        self.val_size, self.val_batch, self.val_wander, self.val_colors = val_size, val_batch, val_wander, val_colors
        self.train_seed, self.val_seed = train_seed, val_seed

    def train_dataloader(self):
        return DataLoader(
                    BrownianAntipodalPaths(self.train_size, self.train_wander, self.train_colors, self.train_seed),
                    batch_size = self.train_batch,
                    shuffle = False,
                    num_workers = 0)

    def val_dataloader(self):
        return DataLoader(
                    BrownianAntipodalPaths(self.val_size, self.val_wander, self.val_colors, self.val_seed),
                    batch_size = self.val_batch,
                    shuffle = False,
                    num_workers = 0)

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
            self.cube.algo([int(i) for i in list(randperm(len(Cube.actions), generator=self.gen, dtype=uint8))][:self.wander])
            self.path = Cube.orbits[int(randint(len(Cube.orbits), (1,1), generator=self.gen, dtype=int16))]
            self.start = self.cube.toColor()
            self.cube.algo(self.path)
            self.finish = self.cube.toColor()
            self.cube.setState(self.start)

        try:
            return tensor(next(self.perms)).float(), one_hot(tensor(self.action),18).float()
        except StopIteration:
            self.action = self.path[0]
            self.path = self.path[1:]
            self.perms = RandomColorPermutor(self.gen, [self.start, self.cube.toColor(), self.finish], self.colors)
            self.cube.act(self.action)
            return tensor(next(self.perms)).float(), one_hot(tensor(self.action),18).float()

    def __len__(self):
        return self.size

class RandomColorPermutor():
    def __init__(self, gen, cubes, limit):
        self.gen = gen
        self.limit = limit
        self.cubes = cubes

    def __next__(self):
        if self.limit < 1:
            raise StopIteration
        self.limit -= 1
        out = [int(i) for i in list(randperm(8, generator=self.gen, dtype=uint8))][:6]
        out = [[out[i] for i in cube] for cube in self.cubes]
        out = [[Cube.color_hot[i] for i in cube] for cube in out]
        out = [item for sublist in out for item in sublist]
        out = [item for sublist in out for item in sublist]
        return out

    def __iter__(self):
        return self

if __name__ == "__main__":
    module = BrownianAntipodalDataModule(10, 10, 0, 10,  1, 1, 0, 1)
    print("train sample:\n", next(iter(module.train_dataloader())))
    print("\nval sample:\n", next(iter(module.val_dataloader())))
