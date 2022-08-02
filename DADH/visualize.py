import numpy as np
from scipy import io
from PIL import Image
import mat73
import h5py
import hdf5storage
import torch
import sys
import gc
import scipy.io as scio

def log_loss_values(alpha, each_tri_i2t, each_tri_t2i, beta, singled_i_ql, singled_t_ql, gamma,
                    singled_loss_adver_feature, singled_loss_adver_hash):
    err_values = []
    for i in range(0, len(each_tri_i2t)):
        weighted_cos_tri = torch.mean(each_tri_i2t[i]) + torch.mean(each_tri_t2i[i])
        loss_quant = singled_i_ql[i] + singled_t_ql[i]
        err = torch.sum(alpha * weighted_cos_tri + \
                                    beta * loss_quant + gamma * (torch.mean(singled_loss_adver_feature[i]) + torch.mean(
            singled_loss_adver_hash[i])))
        err_values.append(err.item())
    return err_values


def give_top_image(query):
    mat_results = io.loadmat('results.mat')
    mat_fall = io.loadmat('mirflickr25k-fall.mat')
    mat_classes = io.loadmat('mirflickr25k-lall.mat')
    results = mat_results['results']
    indices = mat_results['indices']
    pointers = mat_fall['FAll']
    classes = mat_classes['LAll']
    top8 = []
    top8_classes = []
    top8_result = []
    for i in range(0, 8):
        single_class = []

        # Get top 8 retrieved images
        indice = indices[query - 1][0][i]
        top8.append(pointers[indice][0][0])

        # Check if correct retrieval or not
        if results[query - 1][i] == 1:
            top8_result.append("Yes")
        else:
            top8_result.append("No")

        # Get top 8 retrieved images' classes
        for e in range(0, len(classes[indice])):
            if classes[indice][e] == 1:
                single_class.append(e + 1)
        top8_classes.append(single_class)
    return top8, top8_classes, top8_result


def give_query(query):
    mat_fall = io.loadmat('mirflickr25k-fall.mat')
    mat_classes = io.loadmat('mirflickr25k-lall.mat')
    pointers = mat_fall['FAll']
    classes = mat_classes['LAll']
    query_number = pointers[query + 18014][0][0]
    print("Query numeber:", query, "in dataset:", query + 18014)
    tag_number = int(''.join(i for i in query_number if i.isdigit()))
    f = open("./rawdata/mirflickr/meta/tags_raw/tags" + str(tag_number) + ".txt", "r")
    tags = f.readlines()
    tags_string = "Query tags: \n"
    for i in range(0, len(tags)):
        if i % 3 != 0 or i == 0:
            tags[i] = tags[i].replace('\n', ',')
        tags_string = tags_string + tags[i]

    f.close()

    query_classes = []
    for i in range(0, len(classes[query + 18014])):
        if classes[query + 18014][i] == 1:
            query_classes.append(i + 1)

    return tags_string, query_classes


def get_placeholders():
    images = []
    for i in range(1, 9):
        image = Image.open('./placeholders/placeholder' + str(i) + '.jpg')
        images.append(image)
    return images


def scroll_class1(position):
    mat_fall = io.loadmat('mirflickr25k-fall.mat')
    mat_classes = io.loadmat('mirflickr25k-lall.mat')
    pointers = mat_fall['FAll']
    classes = mat_classes['LAll']
    counter = 0
    image = pointers[0][0][0]
    tags_string = "Tag captions: \n"

    for i in range(0, len(classes)):
        if classes[i][0] == 1:
            counter += 1
        if counter == position:
            print("Image:", position, "/ 2578")
            image = pointers[i][0][0]

            tag_number = int(''.join(i for i in image if i.isdigit()))
            f = open("./rawdata/mirflickr/meta/tags_raw/tags" + str(tag_number) + ".txt", "r")
            tags = f.readlines()
            for i in range(0, len(tags)):
                if i % 3 != 0 or i == 0:
                    tags[i] = tags[i].replace('\n', ',')
                tags_string = tags_string + tags[i]

            break
    image_path = "./rawdata/mirflickr/" + str(image)
    return image_path, tags_string


def fill():
    mat_class1_labels = io.loadmat('LAllClassOne.mat')
    mat_classes = io.loadmat('mirflickr25k-lall.mat')
    class1_labels = mat_class1_labels['LAllClassOne']
    classes = mat_classes['LAll']
    counter = 0
    new_list = []
    print(len(class1_labels))
    for i in range(0, len(classes)):
        if classes[i][0] == 0:
            new_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif classes[i][0] == 1 and counter < 2577:
            new_list.append(class1_labels[counter])
            counter += 1
        elif classes[i][0] == 1 and counter == 2577:
            new_list.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    print(len(new_list))
    npclass1 = np.asarray(new_list, dtype=int)
    data = {'LAllClassOne': npclass1}
    io.savemat('LAllClassOneFull.mat', data)


def unpair_data():
    mat_tags = io.loadmat('Cleared-Set/mirflickr25k-yall.mat')
    tags = mat_tags['YAll']
    new_tags = []
    counter = 0
    counter2 = 0
    unpaired_sample = ([0] * 1386)
    unpaired_sample = np.append(unpaired_sample, 1, axis=None)

    for i in range(0, len(tags)):

        if counter >= 95:
            new_tags.append(unpaired_sample)
            counter += 1
            counter2 += 1
            if counter > 99:
                counter = 0
            continue
        else:
            new_tags.append(np.append(tags[i], 0, axis=None))
        counter += 1
        counter2 += 1

    unpaired_tags = np.asarray(new_tags, dtype=int)
    data_tags = {'YAll': unpaired_tags}
    io.savemat('unpaired-data/MIR-Flickr25K/mirflickr25k-yall-unpaired-95-test.mat', data_tags)
    # hdf5storage.write(data_tags, '.', '2nus-wide-tc10-yall-unpaired90.mat', matlab_compatible=True)


def unpair_data_indices():
    indices = np.loadtxt("error_values.txt")
    indices = np.argsort(indices)
    mat_tags = io.loadmat('Cleared-Set/mirflickr25k-yall.mat')
    tags = mat_tags['YAll']
    unpaired_sample = ([0] * 1386)
    unpaired_sample = np.append(unpaired_sample, 1, axis=None)
    final_tags = np.zeros(shape=(20015, 1387))
    unpaired_pos = []
    count = 0

    '''
    # Balanced
    for i in range(0, 10000):
        if i % 2 == 0:
            pos_to_unpair = np.where(indices == i)
            final_tags[pos_to_unpair] = unpaired_sample
            unpaired_pos = np.append(unpaired_pos, pos_to_unpair, axis=None)
    '''
    '''
    # Top 50%
    for i in range(5000, 10000):
        pos_to_unpair = np.where(indices == i)
        final_tags[pos_to_unpair] = unpaired_sample
        unpaired_pos = np.append(unpaired_pos, pos_to_unpair, axis=None)
    '''
    '''
    # Lower 50%
    for i in range(0, 5000):
        pos_to_unpair = np.where(indices == i)
        final_tags[pos_to_unpair] = unpaired_sample
        unpaired_pos = np.append(unpaired_pos, pos_to_unpair, axis=None)
    '''
    '''
    # Percentages
    for i in range(0, 10000):
        if count < 80:
            pos_to_unpair = np.where(indices == i)
            final_tags[pos_to_unpair] = unpaired_sample
            unpaired_pos = np.append(unpaired_pos, pos_to_unpair, axis=None)
        count += 1
        if count == 99:
            count = 0
    '''

    # Unpair all
    for i in range(0, 10000):
        final_tags[i] = unpaired_sample
    for i in range(10000, 20015):
        final_tags[i] = np.append(tags[i], 0, axis=None)

    '''
    # Add rest which have not been unpaired
    for i in range(0, 20015):
        if i not in unpaired_pos:
            final_tags[i] = np.append(tags[i], 0, axis=None)
    '''

    unpaired_tags = np.asarray(final_tags, dtype=int)
    data_tags = {'YAll': unpaired_tags}
    io.savemat('unpaired-data/MIR-Flickr25K/mirflickr25k-yall-unpaired-all.mat', data_tags)
    # hdf5storage.write(data_tags, '.', '2nus-wide-tc10-yall-unpaired90.mat', matlab_compatible=True)

def unpair_data_indices_images():
    indices = np.loadtxt("error_values.txt")
    indices = np.argsort(indices)
    data_file = scio.loadmat("data/FLICKR-25K.mat")
    images = data_file['images'][:]
    print(images.shape)
    unpaired_sample = np.zeros(shape=(3, 224, 224))
    images[0] = unpaired_sample
    final_images = np.zeros(shape=(10000, 3, 224, 224))
    unpaired_pos = []
    count = 0
    #'''
    # Percentages
    for i in range(0, 10000):
        if i % 1000 == 0:
            print("Percentages process:", str(i/10000*100)+"%")
        if count < 80:
            pos_to_unpair = np.where(indices == i)
            final_images[pos_to_unpair] = unpaired_sample
            unpaired_pos = np.append(unpaired_pos, pos_to_unpair, axis=None)
        count += 1
        if count == 99:
            count = 0
    #'''

    '''
    # Unpair all
    for i in range(0, 10000):
        final_tags[i] = unpaired_sample
    for i in range(10000, 20015):
        final_tags[i] = np.append(tags[i], 0, axis=None)
    '''

    #'''
    # Add rest which have not been unpaired
    for i in range(0, 10000):
        if i % 1000 == 0:
            print("Adding process:", str(i/20015*100)+"%")
        if i not in unpaired_pos:
            final_images[i] = images[i]
    #'''

    unpaired_images = np.asarray(final_images, dtype=int)
    unpaired_images = np.transpose(unpaired_images)
    data_tags = {'IAll': unpaired_images}
    del unpaired_images
    #del mat_images
    del images
    del final_images
    gc.collect()
    #mat73.savemat('unpaired-data/MIR-Flickr25K/mirflickr25k-iall-unpaired-80.mat', data_tags)
    #hdf5storage.write(data_tags, '.', 'unpaired-data/MIR-Flickr25K/mirflickr25k-iall-unpaired-80-train.mat', matlab_compatible=True)

def unpair_data_indices_images_new():
    indices = np.loadtxt("error_values.txt")
    indices = np.argsort(indices)
    data_file = scio.loadmat("data/FLICKR-25K.mat")
    images = data_file['images'][:]
    unpaired_sample = ([0] * 4096)
    unpaired_sample = np.append(unpaired_sample, 1, axis=None)
    final_images = np.zeros(shape=(20015, 4097))
    unpaired_pos = []
    count = 0
    #'''
    # Percentages
    for i in range(0, 10000):
        if count < 80:
            pos_to_unpair = np.where(indices == i)
            final_images[pos_to_unpair] = unpaired_sample
            unpaired_pos = np.append(unpaired_pos, pos_to_unpair, axis=None)
        count += 1
        if count == 99:
            count = 0
    #'''

    '''
    # Unpair all
    for i in range(0, 10000):
        final_images[i] = unpaired_sample
    for i in range(10000, 20015):
        final_images[i] = np.append(images[i], 0, axis=None)
    '''

    #'''
    # Add rest which have not been unpaired
    for i in range(0, 20015):
        if i not in unpaired_pos:
            #final_images[i] = images[i]
            final_images[i] = np.append(images[i], 0, axis=None)
    #'''

    unpaired_images = np.asarray(final_images, dtype=int)
    data_images = {'images': unpaired_images}
    io.savemat('unpaired-data/MIR-Flickr25K/mirflickr25k-iall-unpaired-20.mat', data_images)


'''
mat_classes = io.loadmat('LAllClassOneFull.mat')
classes = mat_classes['LAllClassOne']
query = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
retrieval = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
counter = 0
for i in range(0, len(classes)):
    if i < 18015:
        retrieval = retrieval + classes[i]
    else:
        query = query + classes[i]

query_count = 0

for i in range (0, len(query)):
    query_count += query[i]

print("Query count total:", query_count)
print("Query:", query)
print("Retrieval:", retrieval)
print("Total:", query+retrieval)
'''
'''
#my_file = open("error_values.txt", "r")
#content = my_file.read()
content = np.loadtxt('error_values.txt', delimiter='\n')
content = np.argsort(content)
print(content[1300])
content = content / 2000 * 100
#sorted = [x / 2000 for x in sorted]
#sorted = [x * 100 for x in sorted]
'''
'''
with open('percentile_error_values.txt', 'w') as f:
    for item in content:
        f.write("%s\n" % item)
'''
#unpair_data()

#unpair_data_indices()

unpair_data_indices_images_new()
