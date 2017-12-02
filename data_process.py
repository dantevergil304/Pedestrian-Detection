import scipy.io as sio
import cv2
import numpy as np
from os import listdir
from os.path import isfile
import HOGDescriptor

WIN_WIDTH = 64
WIN_HEIGHT = 128


def get_annotation_structure(filename):
    path = 'data/' + filename + '.mat'
    mat_contents = sio.loadmat(path)
    annotations_structure = mat_contents['annotation_struct'].squeeze()
    return annotations_structure


def get_info_annotation(records):
    infos = []
    for record in records:
        filename = record[0][0]
        size = record[1].squeeze()
        objects = record[3][0]
        boundingBoxs = [obj[2].squeeze() for obj in objects]
        infos.append((filename, size, boundingBoxs))
    return infos


def get_image(path):
    for file in listdir(path):
        filepath = path + '/' + file
        if isfile(filepath):
            normalized = cv2.resize(cv2.imread(filepath), (WIN_WIDTH, WIN_HEIGHT))
            yield normalized


def get_data(path):
    hog = HOGDescriptor.createHOGDescriptor((WIN_WIDTH, WIN_HEIGHT), (16, 16), (8, 8), (8, 8), 9, False)
    return np.array([hog.compute(image) for image in get_image(path)]).squeeze()


def get_data_and_label(pos_data_path, neg_data_path):
    pos_image = [image for image in get_image(pos_data_path)]
    neg_image = [image for image in get_image(neg_data_path)]

    pos_data = get_data(pos_data_path)
    neg_data = get_data(neg_data_path)

    nPos = pos_data.shape[0]
    nNeg = neg_data.shape[0]

    pos_label = np.repeat([1], nPos)
    neg_label = np.repeat([0], nNeg)

    image = np.concatenate((pos_image, neg_image))
    data = np.concatenate((pos_data, neg_data))
    label = np.concatenate((pos_label, neg_label))

    rand = np.random.RandomState(9)
    shuffle = rand.permutation(nPos + nNeg)
    image, data, label = image[shuffle], data[shuffle], label[shuffle]

    return image, data, label


def draw_bounding_box(image, bounding_boxs):
    modIm = image.copy()
    for bb in bounding_boxs:
        cv2.rectangle(modIm, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
    return modIm


if __name__ == "__main__":
    '''train_record = get_annotation_structure('train_annotations_structure')
    test_record = get_annotation_structure('test_annotations_structure')
    train_record_info = get_info_annotation(train_record)

    index = 506
    path = train_record_info[index][0]
    image = cv2.imread('data/' + path)
    bounding_box = train_record_info[index][2]

    mod_image = draw_bounding_box(image, bounding_box)
    cv2.imshow('%s' % path, mod_image)'''
    train_pos = 'data/train_64x128_H96/pos'
    train_neg = 'data/train_64x128_H96/neg'
    train_data = get_data(train_pos)
    print(train_data.shape)
