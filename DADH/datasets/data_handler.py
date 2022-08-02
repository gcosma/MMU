import os
import sys

import numpy as np
import h5py
import scipy.io as scio
import mat73
import hdf5storage

def load_data(path, type='flickr25k'):
    if type == 'flickr25k':
        return load_flickr25k(path)
    elif type == 'flickr25kunpaired80balanced':
        return load_flickr25k_unpaired(path)
    elif type == 'flickr25kunpairedimageall':
        return load_flickr25k_unpaired_image(path)
    elif type == 'nus':
        return load_nus_wide(path)
    else:
        return load_nus_wide_unpaired(path)


def load_flickr25k(path):
    data_file = scio.loadmat(path)
    images = data_file['images'][:]
    images = (images - images.mean()) / images.std()
    tags = data_file['YAll'][:]
    labels = data_file['LAll'][:]
    return images, tags, labels


def load_flickr25k_unpaired(path):
    data_file = scio.loadmat(path)
    images = data_file['images'][:]
    images = (images - images.mean()) / images.std()
    data_file_tags = scio.loadmat('unpaired-data/MIR-Flickr25K/mirflickr25k-iall-unpaired-80-balanced.mat')
    tags = data_file_tags['YAll'][:]
    labels = data_file['LAll'][:]
    return images, tags, labels

def load_flickr25k_unpaired_image(path):
    data_file = scio.loadmat(path)
    images_file = scio.loadmat('unpaired-data/MIR-Flickr25K/mirflickr25k-iall-unpaired-all.mat')
    images = images_file['images'][:]
    images = (images - images.mean()) / images.std()
    tags = data_file['YAll'][:]
    labels = data_file['LAll'][:]
    return images, tags, labels

def load_nus_wide(path_dir):
    data_file = scio.loadmat(path_dir)
    images = data_file['image'][:]
    images = (images - images.mean()) / images.std()
    tags = data_file['text'][:]
    labels = data_file['label'][:]

    return images, tags, labels

def load_nus_wide_unpaired(path_dir):
    data_file = scio.loadmat(path_dir)
    images = data_file['image'][:]
    images = (images - images.mean()) / images.std()
    data_file_tags = scio.loadmat('nus-wide-tc21-yall-unpaired50.mat')
    tags = data_file_tags['YAll'][:]
    labels = data_file['label'][:]
    return images, tags, labels



def load_pretrain_model(path):
    return scio.loadmat(path)

