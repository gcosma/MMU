import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.io as scio
import pandas as pd


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


x_array = np.loadtxt('error_values.txt')
normalised_arr = NormalizeData(x_array)
sorted_arr = np.sort(normalised_arr)
balanced_arr = []
for i in range(0, len(normalised_arr)):
    if i % 2 == 0:
        balanced_arr.append(normalised_arr[i])

data = [balanced_arr, sorted_arr[:5000], sorted_arr[5000:]]
fig7, ax7 = plt.subplots()
ax7.set_title('Balanced, bottom 50% and top 50% unpaired sample set loss')
ax7.boxplot(data)

labels = ('Balanced', 'Bottom 50%', 'Top 50%')
plt.xticks(np.arange(len(labels))+1, labels)

print("Standard deviation of sample loss:", np.std(normalised_arr))

plt.show()


def loadmat():
    f = h5py.File('unpaired-data/MIR-Flickr25K/mirflickr25k-iall-unpaired-80-train.mat', 'r')
    data = f.get('IAll')
    images = np.array(data)  # For converting to a NumPy array
    #images = (images - images.mean()) / images.std()
    for i in range(0, len(images)):
        print(images[i].shape)
    return

def load_flickr25k_unpaired_image(path):
    '''
    #data_file = scio.loadmat(path)
    #f = h5py.File('unpaired-data/MIR-Flickr25K/unpaired-images-73.mat', 'r')
    f = h5py.File('unpaired-data/MIR-Flickr25K/mirflickr25k-iall-unpaired-80-train.mat', 'r')
    data_images_train = f.get('IAll')
    images_train = np.array(data_images_train)
    images_train = (images_train - images_train.mean()) / images_train.std()
    data_images_rest = mat73.loadmat('Cleared-Set/mirflickr25k-iall.mat')
    images = data_images_rest['IAll'][10000:]
    images = (images - images.mean()) / images.std()
    data_tags = scio.loadmat('Cleared-Set/mirflickr25k-yall.mat')
    tags = data_tags['YAll'][:]
    data_labels = scio.loadmat('Cleared-Set/mirflickr25k-lall.mat')
    labels = data_labels['LAll'][:]
    '''
    data_file = scio.loadmat(path)
    images = data_file['images'][10000:]
    for i in range(0, len(images)):
        print(images[i].shape)
    images = (images - images.mean()) / images.std()
    f = h5py.File('unpaired-data/MIR-Flickr25K/mirflickr25k-iall-unpaired-80-train.mat', 'r')
    data_images_train = f.get('IAll')
    images_train = np.array(data_images_train)
    for i in range(0, len(images)):
        print(images_train[i].shape)
    images_train = (images_train - images_train.mean()) / images_train.std()
    images = np.append(images_train, images, axis=None)
    return


#load_flickr25k_unpaired_image('./data/FLICKR-25K.mat')

# np.savetxt('normalised_error_values.txt', normalised_arr,fmt='%.8f')
# print(normalised_arr)
