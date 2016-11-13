from utils import load_age_data
from data import AgeDatagen, _preload
from argparse import Namespace
from tqdm import tqdm

args = Namespace(batch_size=32,
                 width=100,
                 height=100,
                 min_size=400,
                 max_size=800,
                 )



data = load_age_data()

z = [t[1] for t in data]
print(sorted(list(set(z))))

# gen = AgeDatagen(args, data)
#
# for _ in tqdm(range(100)):
#     x, y = next(gen)
#     print(x.shape)
#     print(y.shape)