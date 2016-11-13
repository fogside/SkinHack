from batch_generators import AgeGenderBatchGeneratorFolder

reader = AgeGenderBatchGeneratorFolder(10, (100, 100), (80, 80), (50, 50))

x, a, g = reader.get_supervised_batch()

print(x.shape, a.shape, g.shape)

import matplotlib.pyplot as plt

