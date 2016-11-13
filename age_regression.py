import numpy as np

def smooth_probability(y_, delta):
    #Takes labels and makes probability distributions out of them
    #Delta is the width of hat

    alpha = 1 / float(delta)
    shifts = range(-delta, delta + 1)
    hat = [1 - abs(i) * alpha  for i in shifts]

    p_ = np.zeros(y_.shape + (100,))
    for i in range(y_.shape[0]):
        pos = y_[i]
        for j in range(len(shifts)):
            if pos + shifts[j] >= 0 and pos + shifts[j] < 100:
                p_[i, pos + shifts[j]] = hat[j]

    norm = np.sum(p_, 1)
    p_ = p_ / norm.reshape((-1, 1))
    return p_

#print(smooth_probability(np.array([0, 1, 2]), 4))


from batch_generators import AgeGenderBatchGeneratorFolder

gen = AgeGenderBatchGeneratorFolder(100, (100, 100), (50, 50), (50, 50))

print(len(gen.reader.files_list))



