import numpy as np, scipy

def mask_filling(mask, effective=1):
    """
    Filling the body of the mask borders with one's and returns it:

    111111      111111
    1000001     1111111
    10000001 -> 11111111
    0100001     1111111

    Choose effective = 1 ONLY IF its a state true, that marker's borders are very continius
    :return: A 2-D array
    """
    new_mask = np.zeros([mask.shape[0], mask.shape[1]])
    mask = scipy.ndimage.binary_fill_holes(mask).astype(int)
    for i, row in enumerate(mask):
        if sum(row) > 0:
            top_one = np.argmax(row)
            bot_one = len(row) - np.argmax(row[::-1]) - 1
            new_mask[i, top_one:bot_one] = 1
    if (effective):
        return new_mask
    else:
        for i, row in enumerate(mask.T):
            if sum(row) > 0:
                top_one = np.argmax(row)
                bot_one = len(row) - np.argmax(row[::-1]) - 1
                mask[i, top_one:bot_one] = 1
        return new_mask

    def Mask(batch, color_thresholds=[160, 70, 70], effective=1):  # batch(batch_size, height, width, n_channels)
        """
        Calculates a binary mask of the marked area. If the marker wasn't clear enough, borders may be interpolated.
        :return: An 2-D array
        """
        red, green, blue = batch[:, :, :, 0], batch[:, :, :, 1], batch[:, :, :, 2]
        mask = (red > color_thresholds[0]) & (green < color_thresholds[1]) & (blue < color_thresholds[2])
        for i in batch:
            batch[i] = mask_filling(mask[i], effective)
        return mask
