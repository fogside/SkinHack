# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np


class AgeGenderBatchGenerator(object):
    """
    Generates batches of prepared images labeled with age group and gender.
    """
    def __init__(self, X, age, gender, batch_size, patch_shape, stride, random_offset):
        """
        :param X: python list or numpy array of images.
        Each image should be a numpy array of shape (height, width, channels)
        :param age: python list or numpy array of age group labels
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

        self.num_images = X.shape[0]

        self.patch_shape = patch_shape

        self.stride = stride
        self.random_offset = random_offset

        self.image = self.X[self.index]
        self.cur_age = self.age[self.index: self.index + 1]
        self.cur_gender = self.age[self.index: self.index + 1]

    def get_supervised_train_batch(self):
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

    def get_unsupervised_train_batch(self):
        batch_X, _, _ = self.get_supervised_train_batch()

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

        self.num_images = X.shape[0]

        self.patch_shape = patch_shape

        self.stride = stride
        self.random_offset = random_offset

        self.image = self.X[self.index]
        self.segm = self.Y[self.index]

    def get_supervised_train_batch(self):
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

    def get_unsupervised_train_batch(self):
        batch_X, _ = self.get_supervised_train_batch()

        return batch_X
