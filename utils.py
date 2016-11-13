def _start_shell(local_ns=None):
    """Starts interactive ipython shell within local context
        use as `_start_shell(locals())`
    """
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


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


if __name__ == "__main__":
    data = load_age_data('data/')
    train, val = split_data(data)
    print(len(train), len(val))

    print(train[:10])




