from skimage.io import imread
from skimage.transform import resize
import numpy as np
from itertools import cycle
from tqdm import tqdm
import os


def _load_crop(fname, args):
    try:
        img = np.load(fname)
        # pick size
        size = np.random.randint(args.min_size, args.max_size)
        a, b, _ = img.shape
        # pick upper-left corner
        x, y = np.random.randint(0, a - size), np.random.randint(0, b - size)
        patch = img[x: x + size, y: y + size, :]
        return resize(patch, (args.width, args.height))
    except Exception as err:
        print(err, fname)


class WriDatagen:
    def __init__(self, args, dataset):
        self.dataset = dataset
        self.args = args

        self.pointer = cycle(iter(self.dataset))
        self.x = np.zeros((args.batch_size, args.width, args.height, 3), dtype=np.float32)
        self.y = np.zeros((args.batch_size, args.width, args.height), dtype=np.float32)

    def reset(self):
        self.pointer = cycle(iter(self.dataset))

    def next(self):
        self.x.fill(0.0)
        self.y.fill(0.0)

        for i in range(self.args.batch_size):
            sample = next(self.pointer)
            z = _load_crop(sample, self.args)
            self.x[i, ...] = z[:, :, :3]
            self.y[i, ...] = z[:, :, 3]

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
