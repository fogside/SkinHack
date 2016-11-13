# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import os
import matplotlib.image as mpimg
from utils import _start_shell
from keras.preprocessing.image import ImageDataGenerator


class AgeGenderBatchGenerator(object):
    """
    Generates batches of prepared images labeled with age group and gender.
    """
    def __init__(self, X, age, gender, batch_size, patch_shape, stride, random_offset):
        """
        :param X: python list or numpy array of images.
        Each image should be a numpy array of shape (height, width, channels)
        :param age: python list or numpy array of age group or age labels
        Each label should be a numpy array of shape (1,)
        :param gender: python list or numpy array of gender labels
        Each label should be a numpy array of shape (1,)
        :param batch_size: number of samples returned by single call.
        :param patch_shape: tuple, shape of patches sampled from the images (height, width)
        :param stride: tuple, stride step sizes (vertical, horizontal)
        :param random_offset: tuple, maximum values of random initial offsets (vertical, horizontal)
        """

        self.X = X
        self.age = age
        self.gender = gender
        self.batch_size = batch_size

        self.index = 0
        self.offset = np.random.randint(random_offset[1])
        self.i = np.random.randint(random_offset[0])
        self.j = self.offset

        self.num_images = len(X)

        self.patch_shape = patch_shape

        self.stride = stride
        self.random_offset = random_offset

        self.image = self.X[self.index]
        self.cur_age = self.age[self.index: self.index + 1]
        self.cur_gender = self.age[self.index: self.index + 1]

    def get_supervised_batch(self):
        """
        :return: A tuple of numpy arrays: (data_batch, age_labels, gender_labels)
        """
        batch_patches = []
        batch_age = []
        batch_gender = []

        for i in range(self.batch_size):

            patch = self.image[self.i:self.i + self.patch_shape[0], self.j:self.j + self.patch_shape[1]]
            batch_patches.append(np.expand_dims(patch, 0))
            batch_age.append(self.cur_age)
            batch_gender.append(self.cur_gender)

            self.j += self.stride[1]

            if self.j >= self.image.shape[1] - self.patch_shape[1]:

                self.i += self.stride[0]
                self.j = self.offset

                if self.i >= self.image.shape[0] - self.patch_shape[0]:

                    self.i = np.random.randint(self.random_offset[0])
                    self.offset = np.random.randint(self.random_offset[1])
                    self.j = self.offset

                    self.index += 1

                    if self.index >= self.num_images:
                        self.index = 0
                        order = np.random.permutation(self.X.shape[0])
                        self.X = self.X[order]
                        self.age = self.age[order]
                        self.gender = self.gender[order]

                    self.image = self.X[self.index]
                    self.cur_age = self.age[self.index: self.index + 1]
                    self.cur_gender = self.age[self.index: self.index + 1]

        return np.concatenate(batch_patches), np.concatenate(batch_age), np.concatenate(batch_gender)

    def get_unsupervised_batch(self):
        """
        :return: A batch of data as a numpy array
        """
        batch_X, _, _ = self.get_supervised_batch()

        return batch_X


class SegmentationBatchGenerator(object):
    """
    Generates batches of prepared images labeled with age group and gender.
    """
    def __init__(self, X, Y, batch_size, patch_shape, stride, random_offset):
        """
        :param X: python list or numpy array of images.
        Each image should be a numpy array of shape (height, width, channels)
        :param Y: python list or numpy array of segmentation masks.
        Each image should be a numpy array of shape (height, width, channels)
        :param batch_size: number of samples returned by single call.
        :param patch_shape: tuple, shape of patches sampled from the images (height, width)
        :param stride: tuple, stride step sizes (vertical, horizontal)
        :param random_offset: tuple, maximum values of random initial offsets (vertical, horizontal)
        """
        self.X = X
        self.Y = Y
        self.batch_size = batch_size

        self.index = 0
        self.offset = np.random.randint(random_offset[1])
        self.i = np.random.randint(random_offset[0])
        self.j = self.offset

        self.num_images = len(X)

        self.patch_shape = patch_shape

        self.stride = stride
        self.random_offset = random_offset

        self.image = self.X[self.index]
        self.segm = self.Y[self.index]

    def get_supervised_batch(self):
        """
        :return: A tuple of numpy arrays: (data_batch, segmentation_masks)
        """
        batch_patches = []
        batch_segms = []

        for i in range(self.batch_size):

            patch = self.image[self.i:self.i + self.patch_shape[0], self.j:self.j + self.patch_shape[1]]
            batch_patches.append(np.expand_dims(patch, 0))
            segm = self.segm[self.i:self.i + self.patch_shape[0], self.j:self.j + self.patch_shape[1]]
            batch_segms.append(np.expand_dims(segm, 0))

            self.j += self.stride[1]

            if self.j >= self.image.shape[1] - self.patch_shape[1]:

                self.i += self.stride[0]
                self.j = self.offset

                if self.i >= self.image.shape[0] - self.patch_shape[0]:

                    self.i = np.random.randint(self.random_offset[0])
                    self.offset = np.random.randint(self.random_offset[1])
                    self.j = self.offset

                    self.index += 1

                    if self.index >= self.num_images:
                        self.index = 0
                        order = np.random.permutation(self.X.shape[0])
                        self.X = self.X[order]
                        self.Y = self.Y[order]

                    self.image = self.X[self.index]
                    self.segm = self.Y[self.index]

        return np.concatenate(batch_patches), np.concatenate(batch_segms)

    def get_unsupervised_batch(self):
        """
        :return: A batch of data as a numpy array
        """
        batch_X, _ = self.get_supervised_batch()

        return batch_X


class UnsupervisedBatchGenerator(object):
    """
    Generates batches of prepared images without any labels.
    """
    def __init__(self, X, batch_size, patch_shape, stride, random_offset):
        """
        :param X: python list or numpy array of images.
        Each image should be a numpy array of shape (height, width, channels)
        :param batch_size: number of samples returned by single call.
        :param patch_shape: tuple, shape of patches sampled from the images (height, width)
        :param stride: tuple, stride step sizes (vertical, horizontal)
        :param random_offset: tuple, maximum values of random initial offsets (vertical, horizontal)
        """
        self.X = X
        self.batch_size = batch_size

        self.index = 0
        self.offset = np.random.randint(random_offset[1])
        self.i = np.random.randint(random_offset[0])
        self.j = self.offset

        self.num_images = len(X)

        self.patch_shape = patch_shape

        self.stride = stride
        self.random_offset = random_offset

        self.image = self.X[self.index]

    def get_supervised_batch(self):
        raise NotImplementedError()

    def get_unsupervised_batch(self):
        """
        :return: A batch of data as a numpy array
        """
        batch_patches = []

        for i in range(self.batch_size):

            patch = self.image[self.i:self.i + self.patch_shape[0], self.j:self.j + self.patch_shape[1]]
            batch_patches.append(np.expand_dims(patch, 0))

            self.j += self.stride[1]

            if self.j >= self.image.shape[1] - self.patch_shape[1]:

                self.i += self.stride[0]
                self.j = self.offset

                if self.i >= self.image.shape[0] - self.patch_shape[0]:

                    self.i = np.random.randint(self.random_offset[0])
                    self.offset = np.random.randint(self.random_offset[1])
                    self.j = self.offset

                    self.index += 1

                    if self.index >= self.num_images:
                        self.index = 0
                        order = np.random.permutation(self.X.shape[0])
                        self.X = self.X[order]

                    self.image = self.X[self.index]

        return np.concatenate(batch_patches)


class ImageFolderReader(object):
    """
    Cyclic reader of folders of images.
    """
    def __init__(self, path):
        self.buffer_size = 100
        _path = path if path[-1] == '/' else path + '/'

        self.files_list = []
        for (_, _, filenames) in os.walk(_path):
            self.files_list.extend((_path + x for x in filenames))
            break

        self.index = 0

    def read(self, n):
        """
        :param n:
        :return: a list of pictures as numpy arrays of shape (height, width, channels).
        Shapes may differ for different images.
        """
        file_names = []
        for i in range(n):
            file_names.append(self.files_list[self.index])
            self.index = (self.index + 1) % len(self.files_list)

        return list(map(mpimg.imread, file_names))

    def read_all(self):
        return self.read(len(self.files_list))

    def read_by_index(self, index):
        return mpimg.imread(self.files_list[index])


class Augmentor(object):
    def __init__(self, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                 zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'):

        self.image_augmentor = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            fill_mode=fill_mode)

    def __call__(self, image_batch, mask_batch=None):
        state = np.random.get_state()

        aug_image_batch = np.array(image_batch)
        for i in range(image_batch.shape[0]):
            aug_image_batch[i] = self.image_augmentor.random_transform(image_batch[i])

        if mask_batch is not None:
            np.random.set_state(state)
            aug_mask_batch = np.array(mask_batch)
            for i in range(mask_batch.shape[0]):
                aug_mask_batch[i] = self.image_augmentor.random_transform(mask_batch[i])

            return aug_image_batch, aug_mask_batch
        else:
            return aug_image_batch


class TripleBatchGenerator(object):
    """
    Generates batches of prepared images without any labels.
    """

    def __init__(self, X, batch_size, patch_shape, stride, random_offset):
        """
        :param X: python list or numpy array of images.
        Each image should be a numpy array of shape (height, width, channels)
        :param batch_size: number of samples returned by single call.
        :param patch_shape: tuple, shape of patches sampled from the images (height, width)
        :param stride: tuple, stride step sizes (vertical, horizontal)
        :param random_offset: tuple, maximum values of random initial offsets (vertical, horizontal)
        """
        self.X = X
        self.batch_size = batch_size
        self.batch_remainder = batch_size

        self.index = 0
        self.offset = np.random.randint(random_offset[1])
        self.i = np.random.randint(random_offset[0])
        self.j = self.offset

        self.num_images = len(X)

        self.patch_shape = patch_shape

        self.stride = stride
        self.random_offset = random_offset

        self.image = self.X[self.index]
        self.image2 = self.X[self.index + 1]

    """
    Collecting patches to make first pathes in each triple
    """

    def get_patches_from_first_image(self):
        """
        :return: A batch of data as a numpy array
        """
        batch_patches = []

        for i in range(self.batch_size):
            self.batch_remainder -= 1
            patch = self.image[self.i: self.i + self.patch_shape[0], self.j: self.j + self.patch_shape[1]]
            batch_patches.append(np.expand_dims(patch, 0))

            self.j += self.stride[1]

            if self.j >= self.image.shape[1] - self.patch_shape[1]:

                self.i += self.stride[0]
                self.j = self.offset

                if self.i >= self.image.shape[0] - self.patch_shape[0]:

                    self.i = np.random.randint(self.random_offset[0])
                    self.offset = np.random.randint(self.random_offset[1])
                    self.j = self.offset

                    self.index += 1

                    if self.index >= self.num_images:
                        self.index = 0
                        order = np.random.permutation(self.X.shape[0])
                        self.X = self.X[order]

                    self.image = self.X[self.index]
                    return np.concatenate(batch_patches)

            if self.batch_remainder == 0:
                return np.concatenate(batch_patches)

        return np.concatenate(batch_patches)

    """
    Getting random patch from image
    """

    def get_random_patch(self):
        I = np.random.randint(self.image.shape[0] - self.patch_shape[0])
        J = np.random.randint(self.image.shape[1] - self.patch_shape[1])
        patch = self.image2[I: I + self.patch_shape[0], J: J + self.patch_shape[1]]
        return patch

    """
    Making amount of triples
    """

    def get_triple_batch(self):
        """
        :return: A batch of data as a numpy array n x 3
        """
        triple_batch = []

        while (self.batch_remainder > 0):
            first_im_patches = self.get_patches_from_first_image()
            np.random.shuffle(first_im_patches)

            for i in range(first_im_patches.shape[0]):
                j = np.random.randint(self.X.shape[0] - 2)
                if j >= self.index:
                    j += 1
                self.image2 = self.X[j]
                triple_batch.append([first_im_patches[i], first_im_patches[(i + 1) % first_im_patches.shape[0]], \
                                     self.get_random_patch()])
                # print(self.index, j)

        self.batch_remainder = self.batch_size
        return np.array(triple_batch)


class SegmentationBatchGeneratorFolder(object):
    """
    Generates batches of prepared images labeled with age group and gender.
    """

    def __init__(self, image_path, mask_path, batch_size=100, patch_shape=(200, 200), stride=(160, 160),
                 random_offset=(100, 100), buffer_size=100):
        """
        :param pathX: path
        :param image_path:
        :param mask_path: folder with images with masks on them
        Each image should be a numpy array of shape (height, width, channels)
        :param batch_size: number of samples returned by single call.
        :param patch_shape: tuple, shape of patches sampled from the images (height, width)
        :param stride: tuple, stride step sizes (vertical, horizontal)
        :param random_offset: tuple, maximum values of random initial offsets (vertical, horizontal)
        """

        self.reader = ImageSegmentationFolderReader(path=image_path, path_m=mask_path)
        self.X, self.Y = self.reader.read(buffer_size)
        self.Y = self.mask(np.array(self.Y))

        self.batch_size = batch_size
        self.index = 0
        self.offset = np.random.randint(random_offset[1])
        self.i = np.random.randint(random_offset[0])
        self.j = self.offset

        self.patch_shape = patch_shape

        self.stride = stride
        self.random_offset = random_offset

        self.image = self.X[self.index]
        self.segm = self.Y[self.index]

        self.buffer_size = buffer_size

    def mask(self, btch, color_thresholds=(220, 20, 20)):  # batch(batch_size, height, width, n_channels)
        """
        Calculates a binary mask of the marked area. If the marker wasn't clear enough, borders may be interpolated.
        :return: An 4-D array of shape (batch_size, height, width, n_channels)
        """

        red, green, blue = btch[:, :, :, 0], btch[:, :, :, 1], btch[:, :, :, 2]
        mask = (red > color_thresholds[0]) & (green < color_thresholds[1]) & (blue < color_thresholds[2])
        mask = mask.astype(float)
        mask = mask.reshape([btch.shape[0], btch.shape[1], btch.shape[2], 1])
        return mask

    def get_supervised_batch(self):
        """
        :return: A tuple of numpy arrays: (data_batch, segmentation_masks)
        """
        batch_patches = []
        batch_segms = []

        for i in range(self.batch_size):
            patch = self.image[self.i:self.i + self.patch_shape[0], self.j:self.j + self.patch_shape[1]]
            batch_patches.append(np.expand_dims(patch, 0))
            segm = self.segm[self.i:self.i + self.patch_shape[0], self.j:self.j + self.patch_shape[1]]
            batch_segms.append(np.expand_dims(segm, 0))

            self.j += self.stride[1]

            if self.j >= self.image.shape[1] - self.patch_shape[1] - 10:

                self.i += self.stride[0]
                self.j = self.offset

                if self.i >= self.image.shape[0] - self.patch_shape[0] - 10:

                    self.i = np.random.randint(self.random_offset[0])
                    self.offset = np.random.randint(self.random_offset[1])
                    self.j = self.offset

                    self.index += 1

                    if self.index >= self.buffer_size:
                        self.index = 0
                        self.X, self.Y = self.reader.read(self.buffer_size)

                    self.image = self.X[self.index]
                    self.segm = self.Y[self.index]
        return np.concatenate(batch_patches), np.concatenate(batch_segms)

    def get_unsupervised_batch(self):
        """
        :return: A batch of data as a numpy array
        """
        batch_X, _ = self.get_supervised_batch()

        return batch_X


class ImageSegmentationFolderReader(object):
    """
    Cyclic reader of folders of images.
    """
    def __init__(self, path_m, path):
        self.buffer_size = 100
        _path = path if path[-1] == '/' else path + '/'
        _path_m = path_m if path_m[-1] == '/' else path_m + '/'
        self.files_list = []
        self.files_list_m = []
        for (_, _, filenames) in os.walk(_path):
            self.files_list.extend((_path + x for x in filenames))
            self.files_list_m.extend((_path_m + self.add(x) for x in filenames))
            break
        self.index = 0

    def add(self, path):
        ind = path.find('.')
        # if names of files are same in both folders delete _m ---------------------------------------------------------
        path = path[:ind] + "_m" + path[ind:]
        return path

    def read(self, n):
        """
        :param n:
        :return: a list of pictures as numpy arrays of shape (height, width, channels).
        Shapes may differ for different images.
        """
        file_names = []
        file_names_m = []
        for i in range(n):
            file_names.append(self.files_list[self.index])
            file_names_m.append(self.files_list_m[self.index])
            self.index = (self.index + 1) % len(self.files_list)
            if self.index == len(self.files_list):
                self.index = 0
                state = np.random.get_state()
                np.random.shuffle(file_names)
                np.random.set_state(state)
                np.random.shuffle(file_names_m)

        X, Y = list(map(mpimg.imread, file_names)), list(map(mpimg.imread, file_names_m))
        return X, Y


    def read_all(self):
        return self.read(len(self.files_list))

class UnsupervisedBatchGeneratorFolder(object):
    """
    Generates batches of prepared images labeled with age group and gender.
    """

    def __init__(self, number_of_images, image_path, batch_size=100, patch_shape=(200, 200), stride=(160, 160), random_offset=(100, 100)):
        """
        :param pathX: path
        :param image_path:
        Each image should be a numpy array of shape (height, width, channels)
        :param batch_size: number of samples returned by single call.
        :param patch_shape: tuple, shape of patches sampled from the images (height, width)
        :param stride: tuple, stride step sizes (vertical, horizontal)
        :param random_offset: tuple, maximum values of random initial offsets (vertical, horizontal)
        """

        self.reader = ImageUnsupervisedFolderReader(path=image_path)
        self.X = self.reader.read(number_of_images)
        self.batch_size = batch_size
        self.index = 0
        self.offset = np.random.randint(random_offset[1])
        self.i = np.random.randint(random_offset[0])
        self.j = self.offset

        self.num_images = len(self.X)

        self.patch_shape = patch_shape

        self.stride = stride
        self.random_offset = random_offset

        self.image = self.X[self.index]

    def mask(self, btch, color_thresholds=(220, 20, 20)):  # batch(batch_size, height, width, n_channels)
        """
        Calculates a binary mask of the marked area. If the marker wasn't clear enough, borders may be interpolated.
        :return: An 4-D array of shape (batch_size, height, width, n_channels)
        """
        red, green, blue = btch[:, :, :, 0], btch[:, :, :, 1], btch[:, :, :, 2]
        mask = (red > color_thresholds[0]) & (green < color_thresholds[1]) & (blue < color_thresholds[2])
        mask = mask.astype(int)
        mask = mask.reshape([btch.shape[0], btch.shape[1], btch.shape[2], 1])
        return mask

    def get_unsupervised_batch(self):
        """
        :return: A tuple of numpy arrays: (data_batch)
        """
        batch_patches = []

        for i in range(self.batch_size):
            patch = self.image[self.i:self.i + self.patch_shape[0], self.j:self.j + self.patch_shape[1]]
            batch_patches.append(np.expand_dims(patch, 0))

            self.j += self.stride[1]

            if self.j >= self.image.shape[1] - self.patch_shape[1]:

                self.i += self.stride[0]
                self.j = self.offset

                if self.i >= self.image.shape[0] - self.patch_shape[0]:

                    self.i = np.random.randint(self.random_offset[0])
                    self.offset = np.random.randint(self.random_offset[1])
                    self.j = self.offset

                    self.index += 1

                    if self.index >= self.num_images:
                        self.index = 0
                        order = np.random.permutation(len(self.X))
                        self.X = self.X[order]


                    self.image = self.X[self.index]


        return np.concatenate(batch_patches)

    def get_supervised_batch(self):
        raise NotImplementedError


class ImageUnsupervisedFolderReader(object):
    """
    Cyclic reader of folders of images.
    """
    def __init__(self, path):
        self.buffer_size = 100
        _path = path if path[-1] == '/' else path + '/'
        self.files_list = []
        for (_, _, filenames) in os.walk(_path):
            self.files_list.extend((_path + x for x in filenames))
            break
        self.index = 0

    def add(self, path):
        ind = path.find('.')
        # if names of files are same in both folders delete _m ---------------------------------------------------------
        path = path[:ind] + "_m" + path[ind:]
        return path

    def read(self, n):
        """
        :param n:
        :return: a list of pictures as numpy arrays of shape (height, width, channels).
        Shapes may differ for different images.
        """
        file_names = []
        file_names_m = []
        for i in range(n):
            file_names.append(self.files_list[self.index])
            self.index = (self.index + 1) % len(self.files_list)
        X = list(map(mpimg.imread, file_names))
        return X


def test():
    assert os.path.exists('data_w/package_1')
    assert os.path.exists("data_w/mapped_1")
    gen = SegmentationBatchGeneratorFolder(10, "data_w/package_1", "data_w/mapped_1", 10, (24, 24), (160, 160), (100, 100))
    aug = Augmentor()
    x, y = gen.get_supervised_batch()
    assert x.shape == (10, 24, 24, 3)
    gen = UnsupervisedBatchGeneratorFolder(10, "data_w/package_1", 10, (24, 24), (160, 160), (100, 100))
    x = gen.get_unsupervised_batch()
    assert x.shape == (10, 24, 24, 3)


class TripleBatchGeneratorFolder(object):
    """
    Generates batches of prepared images without any labels.
    """

    def __init__(self, path, batch_size, patch_shape, stride, random_offset):
        """
        :param path: path to folder containig images
        Each image should be a numpy array of shape (height, width, channels)
        :param batch_size: number of samples returned by single call.
        :param patch_shape: tuple, shape of patches sampled from the images (height, width)
        :param stride: tuple, stride step sizes (vertical, horizontal)
        :param random_offset: tuple, maximum values of random initial offsets (vertical, horizontal)
        """
        self.files_list = ImageFolderReader(path)
        self.batch_size = batch_size
        self.batch_remainder = batch_size

        self.index = 0
        self.offset = np.random.randint(random_offset[1])
        self.i = np.random.randint(random_offset[0])
        self.j = self.offset

        self.num_images = len(self.files_list.files_list)

        self.patch_shape = patch_shape

        self.stride = stride
        self.random_offset = random_offset

        self.image, image2  = self.files_list.read(2)

    """
    Collecting patches to make first pathes in each triple
    """

    def get_patches_from_first_image(self):
        """
        :return: A batch of data as a numpy array
        """
        batch_patches = []

        for i in range(self.batch_size):
            self.batch_remainder -= 1
            patch = self.image[self.i: self.i + self.patch_shape[0], self.j: self.j + self.patch_shape[1]]
            #print(patch.shape)
            batch_patches.append(np.expand_dims(patch, 0))

            self.j += self.stride[1]

            if self.j >= self.image.shape[1] - self.patch_shape[1]:

                self.i += self.stride[0]
                self.j = self.offset

                if self.i >= self.image.shape[0] - self.patch_shape[0]:

                    self.i = np.random.randint(self.random_offset[0])
                    self.offset = np.random.randint(self.random_offset[1])
                    self.j = self.offset

                    self.index += 1

                    if self.index >= self.num_images:
                        self.index = 0

                    self.image = self.files_list.read(1)[0]
                    return np.concatenate(batch_patches)

            if self.batch_remainder == 0:
                return np.concatenate(batch_patches)

        return np.concatenate(batch_patches)

    """
    Getting random patch from image
    """

    def get_random_patch(self):
        I = np.random.randint(self.image.shape[0] - self.patch_shape[0])
        J = np.random.randint(self.image.shape[1] - self.patch_shape[1])
        patch = self.image2[I: I + self.patch_shape[0], J: J + self.patch_shape[1]]
        return patch

    """
    Making amount of triples
    """

    def get_triple_batch(self):
        """
        :return: A batch of data as a numpy array n x 3
        """
        triple_batch = []

        while self.batch_remainder > 0:
            first_im_patches = self.get_patches_from_first_image()
            np.random.shuffle(first_im_patches)

            for i in range(first_im_patches.shape[0]):
                j = np.random.randint(len(self.files_list.files_list) - 2)
                if j >= self.index:
                    j += 1
                self.image2 = self.files_list.read_by_index(j)
                triple_batch.append([first_im_patches[i], first_im_patches[(i + 1) % first_im_patches.shape[0]], \
                                     self.get_random_patch()])
                # print(self.index, j)

        self.batch_remainder = self.batch_size
        #print ("shape of random patch", self.get_random_patch().shape)
        #print ("shape of triple batch", np.array(triple_batch).shape)
        return np.array(triple_batch)

"""
Example^

reader = ImageFolderReader('data/first_data')
import matplotlib.pyplot as plt

gen = UnsupervisedBatchGenerator(reader.read_all(), 100, (200, 200), (160, 160), (100, 100))
aug = Augmentor()
x = aug(gen.get_unsupervised_batch())

plt.imshow(x[5])
plt.show()
"""

import re

class AgeFolderReader(object):
    """
    Cyclic reader of folders of images.
    """
    def __init__(self):
        path = 'data/ddp/'

        self.files_list = []
        for (_, _, filenames) in os.walk(path):
            self.files_list.extend((path + x for x in filenames))
            break

        r = re.compile('\d+')

        self.labels_age = [None] * len(self.files_list)
        self.labels_gender = [None] * len(self.files_list)

        def digits(s):
            return r.findall(s)[0]

        d = dict(((digits(self.files_list[i]), i) for i in range(len(self.files_list))))

        labels = open('data/train_age_full.txt')

        for s in labels:
            fname, age, gend = s.split()
            try:
                fname = digits(fname)
            except:
                continue
            try:
                f_index = d[fname]
                self.labels_age[f_index] = int(age)
                self.labels_gender[f_index] = 0 if gend == 'w' else 1
            except:
                pass

        for i in range(len(self.files_list) - 1, -1, -1):
            if self.labels_age[i] is None or self.labels_gender[i] is None:
                del self.labels_age[i], self.labels_gender[i], self.files_list[i]

        self.labels_age = np.array(self.labels_age)
        self.labels_gender = np.array(self.labels_gender)
        self.index = 0

    def read(self, n):
        """
        :param n:
        :return: a list of pictures as numpy arrays of shape (height, width, channels).
        Shapes may differ for different images.
        """
        file_names = []
        labels_age = []
        labels_gender = []
        for i in range(n):
            file_names.append(self.files_list[self.index])
            labels_age.append(self.labels_age[self.index])
            labels_gender.append(self.labels_gender[self.index])
            self.index = (self.index + 1)
            if self.index == len(self.files_list):
                self.index = 0
                state = np.random.get_state()
                np.random.shuffle(file_names)
                np.random.set_state(state)
                np.random.shuffle(labels_age)
                np.random.set_state(state)
                np.random.shuffle(labels_gender)

        return list(map(mpimg.imread, file_names)), \
               np.array(labels_age).reshape((-1, 1)), \
               np.array(labels_gender).reshape((-1, 1))

    def read_all(self):
        return self.read(len(self.files_list))


class AgeGenderBatchGeneratorFolder(object):
    """
    Generates batches of prepared images labeled with age group and gender.
    """
    def __init__(self, batch_size, patch_shape, stride, random_offset, buffer_size=100, reuse_buffer=4):
        """
        :param X: python list or numpy array of images.
        Each image should be a numpy array of shape (height, width, channels)
        :param age: python list or numpy array of age group or age labels
        Each label should be a numpy array of shape (1,)
        :param gender: python list or numpy array of gender labels
        Each label should be a numpy array of shape (1,)
        :param batch_size: number of samples returned by single call.
        :param patch_shape: tuple, shape of patches sampled from the images (height, width)
        :param stride: tuple, stride step sizes (vertical, horizontal)
        :param random_offset: tuple, maximum values of random initial offsets (vertical, horizontal)
        """
        self.reader = AgeFolderReader()

        self.X, self.age, self.gender = self.reader.read(buffer_size)
        self.batch_size = batch_size
        self.augmentor = Augmentor(rotation_range=0, width_shift_range=0., height_shift_range=0.,
                                   shear_range=0, zoom_range=0, fill_mode='reflect', )

        self.index = 0
        self.offset = np.random.randint(random_offset[1])
        self.i = np.random.randint(random_offset[0])
        self.j = self.offset

        self.num_images = len(self.X)

        self.patch_shape = patch_shape

        self.stride = stride
        self.random_offset = random_offset

        self.image = self.X[self.index] / 255.
        self.cur_age = self.age[self.index: self.index + 1]
        self.cur_gender = self.age[self.index: self.index + 1]
        self.buffer_size = buffer_size
        self.reuse_buffer = reuse_buffer
        self.reuse_count = 0

    def get_supervised_batch(self):
        """
        :return: A tuple of numpy arrays: (data_batch, age_labels, gender_labels)
        """
        batch_patches = []
        batch_age = []
        batch_gender = []

        for i in range(self.batch_size):
            patch = self.image[self.i:self.i + self.patch_shape[0], self.j:self.j + self.patch_shape[1]]
            batch_patches.append(np.expand_dims(patch, 0))
            batch_age.append(self.cur_age)
            batch_gender.append(self.cur_gender)

            self.j += self.stride[1]

            if self.j >= self.image.shape[1] - self.patch_shape[1]:

                self.i += self.stride[0]
                self.j = self.offset

                if self.i >= self.image.shape[0] - self.patch_shape[0]:

                    self.i = np.random.randint(self.random_offset[0])
                    self.offset = np.random.randint(self.random_offset[1])
                    self.j = self.offset

                    self.index += 1

                    if self.index >= self.num_images:
                        self.index = 0
                        self.reuse_count += 1
                        if self.reuse_count == self.reuse_buffer:
                            self.X, self.age, self.gender = self.reader.read(self.buffer_size)
                        else:
                            state = np.random.get_state()
                            np.random.shuffle(self.X)
                            np.random.set_state(state)
                            np.random.shuffle(self.age)
                            np.random.set_state(state)
                            np.random.shuffle(self.gender)

                    self.image = self.X[self.index] / 255.
                    self.cur_age = self.age[self.index: self.index + 1]
                    self.cur_gender = self.age[self.index: self.index + 1]

        return self.augmentor(np.concatenate(batch_patches)), np.concatenate(batch_age), np.concatenate(batch_gender)

    def get_unsupervised_batch(self):
        """
        :return: A batch of data as a numpy array
        """
        batch_X, _, _ = self.get_supervised_batch()

        return batch_X
