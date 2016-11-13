import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from itertools import cycle
from tqdm import tqdm
from argparse import Namespace
import random
from joblib import Parallel, delayed


def _load_brick(img_path, masked_path, outname, width=2200, height=1500):
    R, G, B = 160, 70, 70
    try:
        img = imread(img_path)
        masked = imread(masked_path)

        # okay, this works
        mask = ((masked[:, :, 0] > R) & (masked[:,:,1] < G) & (masked[:,:,2] < B))
        img = resize(img, (height, width))
        mask = resize(img, (height, width))

        # ad hoc, need to 4 channel image (R,G,B,Mask)
        z = np.concatenate([img, mask], axis=-1)
        np.save(outname, z[:, :, :4])

        return outname

    except Exception as err:
        print(err)


def _gen_brick(fin, args):
    img = np.load(fin)

    # we also can pick size
    size = args.size
    a, b, _ = img.shape
    x, y = np.random.randint(0, a - size), np.random.randint(0, b - size)
    patch = img[x: x + size, y: y + size, :]
    # here can we add mask smoothing
    return resize(patch, (args.width, args.height))


def preprocessing(pairs, nb=4):
    """Generate with MP input data
    """
    generated = Parallel(n_jobs=nb, verbose=5)(delayed(_load_brick)(*p) for p in pairs)
    return generated


def generate_brick(flist, outname, args, nb=4):
    """Generate brick for network feeding
    Args:
        list of npy (RGBW) images
        name for output brick
        args -- Namespace with params
        nb -- number of workers
    """
    it = cycle(iter(flist))
    brick = Parallel(n_jobs=nb, verbose=5)(delayed(_gen_brick)(next(it), args) for _ in range(args.samples))
    np.save(outname, brick)


def main():
    args = Namespace(samples=1000, width=128, height=128, size=256)
    tasks = []
    bad = []
    outpath = 'data/ynet-wrinkles/'
    os.makedirs(outpath)

    for path in ['data/wrinkles/package_1', 'data/wrinkles/package_2']:
        for p in os.listdir(path):
            image_path = os.path.join(path, p)
            masked_path = os.path.join(path.replace('package', 'mapped'), p.replace('.jpg', '_m.jpg'))
            if os.path.exists(image_path) and os.path.exists(masked_path):
                out = os.path.join(outpath, p + '.npy')
                tasks.append((image_path, masked_path, out))
            else:
                bad.append((image_path, masked_path))

    print('There are {} valid pairs and {} invalid'.format(len(tasks), len(bad)))

    tasks = tasks[:10]

    files = preprocessing(tasks)

    files.sort()
    random.seed(124)
    random.shuffle(files)
    N = int(0.2 * len(files))
    train_files, val_files = files[:-N], files[-N:]

    print('Generate train')
    generate_brick(train_files, 'data/ynet.train', args)
    print('Generate val')
    generate_brick(val_files, 'data/ynet.val', args)

    print('Done')


if __name__ == "__main__":
    main()
