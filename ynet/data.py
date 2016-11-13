from skimage.io import imread
from skimage.transform import resize
import numpy as np
from itertools import cycle
from tqdm import tqdm
import os


class WriDatagen:
    def __init__(self, args, path):
        self.dataset = np.load(path)
        self.args = args

        self.pointer = cycle(iter(range(len(self.dataset))))
        self.x = np.zeros((args.batch_size, args.width, args.height, 3), dtype=np.float32)
        self.y = np.zeros((args.batch_size, args.width, args.height), dtype=np.float32)

    def reset(self):
        self.pointer = cycle(iter(range(len(self.dataset))))

    def next(self):
        self.x.fill(0.0)
        self.y.fill(0.0)

        for i in range(self.args.batch_size):
            sample = next(self.pointer)
            self.x[i, ...] = self.dataset[sample, :, :, :3]
            self.y[i, ...] = self.dataset[sample, :, :, 3]

        return self.x, self.y

    __next__ = next

    def __iter__(self):
        return self


if __name__ == "__main__":
    from argparse import Namespace
    import utils
    args = Namespace(batch_size=32, samples=3200, width=100, height=100, min_size=200, max_size=800)
    path = '../data/ynet-wrinkles'
    flist = [os.path.join(path, f) for f in os.listdir(path)]

    datagen = WriDatagen(args, flist)

    for _ in tqdm(range(100)):
        next(datagen)
