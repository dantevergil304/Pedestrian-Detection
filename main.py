from helpers import sliding_window
from helpers import pyramid
import numpy as np
import cv2
import nms
import HOGDescriptor
from data_process import get_data, get_data_and_label
from sklearn import svm
import pickle
from svmClassifier import evaluate

if __name__ == "__main__":
    # TRAIN MODEL
    _, train_data, train_label = get_data_and_label('data/train_64x128_H96/pos', 'data/train_64x128_H96/neg')
    model = svm.SVC(probability=True)
    model.fit(train_data, train_label)
    pickle.dump(model, open('modelSK1.xml', 'wb'))

    # TEST MODEL
    model = pickle.load(open('modelSK.xml', 'rb'))
    # test_image, test_data, test_label = get_data_and_label('data/test_64x128_H96/pos', 'data/test_64x128_H96/neg')
    # evaluate(model, test_data, test_label, test_image)

    # DETECT PEDESTRIAN
    hog = HOGDescriptor.createHOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9, False)
    origin_image = cv2.imread('data/Train/pos/crop001048.png')
    origin_r, origin_c, _ = origin_image.shape
    ratio = origin_c / origin_r

    image = cv2.resize(origin_image, None, fx=250/origin_c, fy=250/(ratio * origin_r))
    r, c, _ = image.shape
    print(r, c)
    bounding_box = []
    prob = []
    for (i, layer) in enumerate(pyramid(image, 1.05, (64, 128))):
        m, n, _ = layer.shape
        for (x, y, window) in sliding_window(layer, 8, (64, 128)):
            if window.shape[0] != 128 or window.shape[1] != 64:
                continue
            normalized = cv2.resize(window, (64, 128))
            data = np.array([hog.compute(normalized)]).squeeze()
            data = data.reshape(1, -1)
            resp = model.predict_proba(data)

            x_begin = int((x / n) * c)
            y_begin = int((y / m) * r)
            x_end = int(((x + 63) / n) * c)
            y_end = int(((y + 127) / m) * r)
            if resp[0][1] > resp[0][0]:
                bounding_box.append((x_begin, y_begin, x_end, y_end))
                prob.append(resp[0][1])

    mod_im = image.copy()
    for rect in nms.non_max_suppression_slow(np.array(bounding_box), prob, overlap_thresh=0.27):
        cv2.rectangle(mod_im, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
    cv2.imshow('figure', mod_im)
    cv2.waitKey(0)
