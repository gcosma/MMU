import os
import numpy as np
import h5py
import scipy.io as scio


def preprocess(x, mean, std):
    mean, std = np.array(mean), np.array(std)
    return (x - mean.reshape(3, 1, 1)) / (std.reshape(3, 1, 1) + 1e-5)


def load_data(path, type='flickr25k'):
    if type == 'flickr25k':
        return load_flickr25k(path)
    else:
        return load_nus_wide(path)


def load_flickr25k(path):
    image_file = h5py.File(os.path.join(path, 'mirflickr25k-iall.mat'))
    images = image_file['IAll'][:]
    image_file.close()

    tag_data = scio.loadmat(os.path.join(path, 'mirflickr25k-yall-unpaired-all.mat'))
    tags = tag_data['YAll']

    label_data = scio.loadmat(os.path.join(path, 'mirflickr25k-lall.mat'))
    labels = label_data['LAll']

    '''
    data_file = h5py.File(path)
    images = data_file['IAll'][:]
    print("Length images")
    print(len(images))
    tags = data_file['YAll'][:]
    print("Length tags")
    print(len(tags))
    labels = data_file['LAll'][:]
    data_file.close()
    '''
    return images, tags, labels


def load_nus_wide(path_dir):
    image_file = h5py.File(os.path.join(path_dir, 'nus-wide-tc21-iall.mat'))
    images = image_file['IAll'][:]
    image_file.close()

    tag_data = scio.loadmat(os.path.join(path_dir, 'nus-wide-tc21-yall-unpaired50.mat'))
    tags = tag_data['YAll']

    label_data = scio.loadmat(os.path.join(path_dir, 'nus-wide-tc21-lall.mat'))
    labels = label_data['LAll']

    return images, tags, labels


def data_enhance(images, tags, labels, c=0.5):
    num = images.shape[0]
    ind1 = np.random.permutation(num // 2)
    ind2 = np.random.permutation(num // 2)
    inhanced_imgs = c * images[ind1] + (1 - c) * images[ind2]
    inhanced_tags = c * tags[ind1] + (1 - c) * tags[ind2]
    inhanced_labels = c * labels[ind1] + (1 - c) * labels[ind2]
    new_imgs = np.concatenate((images, inhanced_imgs))
    new_tags = np.concatenate((tags, inhanced_tags))
    new_labels = np.concatenate((labels, inhanced_labels))
    return new_imgs, new_tags, new_labels


def load_pretrain_model(path):
    return scio.loadmat(path)

