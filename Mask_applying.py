def Mask(batch, color_thresholds=[160, 70, 70]):  # batch(batch_size, height, width, n_channels)
    """
    Calculates a binary mask of the marked area. If the marker wasn't clear enough, borders may be interpolated.
    :return: An 2-D array
    """
    red, green, blue = batch[:, :, :, 0], batch[:, :, :, 1], batch[:, :, :, 2]
    mask = (red > color_thresholds[0]) & (green < color_thresholds[1]) & (blue < color_thresholds[2])
    mask = mask.astype(int)
    mask = mask.reshape([batch.shape[0], batch.shape[1], batch.shape[2], 1])
    return mask
