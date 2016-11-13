import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from itertools import cycle
from tqdm import tqdm
from argparse import Namespace
import random
from joblib import Parallel, delayed



def load_age_data(path='data/', drop_na=True):
    """Loads age file lists

    Returns:
        list of (prefix, age_int, sex_int, file_path)
    """
    import os
    from tqdm import tqdm
    """Handle data/data splitting for this fine event

        Returns
         train/val data split. Data in a proper format for generator
    """
    subs = {'w': 1, 'm': -1, 'NA': 0}
    D = {}
    index_file = os.path.join(path, 'train_age_full.txt')
    with open(index_file, 'r') as fin:
        for line in fin:
            try:
                fname, age, sex = line.replace('\t', ' ').replace('\n', '').split(' ')
                if drop_na and sex == 'NA':
                    continue
                fname = fname[:fname.index('.')]
                D[fname] = [int(age), subs[sex]]
            except Exception as err:
                print('!', line)

    bad = []
    data_dir = os.path.join(path, 'ddp')
    files = os.listdir(data_dir)
    for f in tqdm(files):
        i = f.index('.')
        prefix = f[:i]
        full_path = os.path.join(data_dir, f)
        if prefix in D:
            D[prefix].append(full_path)
        else:
            bad.append(full_path)
    print('There are {} photos without info.'.format(len(bad)))

    clean = []
    for k, v in D.items():
        if len(v) == 3:
            clean.append(tuple([k] + v))

    print('There are {} clean samples with photo and info'.format(len(clean)))

    # Repro style
    clean.sort()
    return clean


def split_data(dataset, p=0.2, seed=2512):
    import random
    random.seed(seed)
    random.shuffle(dataset)
    N = int(p * len(dataset))

    return dataset[:-N], dataset[-N:]


def _gen_brick(task, args):
    try:
        _, age, _, fin = task
        img = imread(fin)

        # we also can pick size
        size = np.random.randint(args.min_size, args.max_size)
        a, b, _ = img.shape
        x, y = np.random.randint(0, a - size), np.random.randint(0, b - size)
        patch = img[x: x + size, y: y + size, :]
        # here can we add mask smoothing
        return (resize(patch, (args.width, args.height)), age, True)
    except Exception as err:
        print(err)
        return (0, 0, False)


def generate_brick(flist, outname, args, nb=4):
    """Generate brick for network feeding
    Args:
        list of npy (RGBW) images
        name for output brick
        args -- Namespace with params
        nb -- number of workers
    """
    it = cycle(iter(flist))
    data = Parallel(n_jobs=nb, verbose=5)(delayed(_gen_brick)(next(it), args) for _ in range(args.samples))

    data = [t for t in data if t[-1]]
    x, y = [], []

    for t in data:
        x.append(t[0])
        y.append(t[1])


    x = np.array(x)
    y = np.array(y)
    np.save(outname + 'x.npy', x)
    np.save(outname + 'y.npy', y)


def main():
    args = Namespace(samples=32000, width=128, height=128, min_size=256, max_size=1024)

    data = load_age_data('data/')
    train, val = split_data(data)
    print(len(train), len(val))

    print('Generate train brick')
    generate_brick(train, 'data/agenet.train', args, nb=6)
    print('Generate val brick')
    args.samples = 3200
    generate_brick(val, 'data/agenet.val', args, nb=6)

if __name__ == "__main__":
    main()
