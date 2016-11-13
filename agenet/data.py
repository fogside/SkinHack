from skimage.io import imread
from skimage.transform import resize
import numpy as np
from itertools import cycle
from tqdm import tqdm


def _load_crop(fname, args):
    try:
        img = imread(fname)
        # pick size
        size = np.random.randint(args.min_size, args.max_size)
        a, b, _ = img.shape
        # pick upper-left corner
        x, y = np.random.randint(0, a - size), np.random.randint(0, b - size)
        patch = img[x: x + size, y: y + size]
        return resize(patch, (args.width, args.height))
    except Exception as err:
        print(err, fname)
        return np.empty(args.width, args.height, 3)


def preprocessing(flist, args, seed=5122, nb=4):
    """Generate with MP input data
    """
    from joblib import Parallel, delayed

    np.random.seed(seed)
    it = cycle(iter(flist))

    ret = Parallel(n_jobs=nb, verbose=5)(delayed(_load_crop)(next(it), args) for _ in range(args.samples))
    brick = np.array(ret)

    return brick


    # for i in tqdm(range(args.samples)):
    #     fname = next(pointer)
    #     brick[i, ...] = _load_crop(fname, args)

    # return brick


class AgeDatagen:
    def __init__(self, args, dataset):
        self.dataset = dataset
        self.args = args

        self.pointer = cycle(iter(self.dataset))
        self.x = np.zeros((args.batch_size, args.width, args.height, 3), dtype=np.float32)
        self.y = np.zeros((args.batch_size, args.bins), dtype=np.float32)

    def reset(self):
        self.pointer = cycle(iter(self.dataset))

    def next(self):
        self.x.fill(0.0)
        self.y.fill(0.0)

        for i in range(self.args.batch_size):
            sample = next(self.pointer)
            self.y[i, sample[1] - 18] = 1.0
            self.x[i, ...] = _load_crop(sample[-1], self.args)

        return self.x, self.y

    __next__ = next

    def __iter__(self):
        return self


if __name__ == "__main__":
    from argparse import Namespace
    import utils
    args = Namespace(samples=3200, width=100, height=100, min_size=200, max_size=800)
    data = utils.load_age_data()
    flist = [t[-1] for t in data]

    preprocessing(flist, args)