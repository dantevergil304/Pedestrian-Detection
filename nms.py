import numpy as np


def non_max_suppression_slow(boxes, prob=None, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = ((x2 - x1 + 1) * (y2 - y1 + 1)).astype("float")

    index = y2
    if prob is not None:
        index = prob

    index = np.argsort(index)

    while len(index) > 0:
        last = len(index) - 1
        i = index[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[index[:last]])
        yy1 = np.maximum(y1[i], y1[index[:last]])
        xx2 = np.minimum(x2[i], x2[index[:last]])
        yy2 = np.minimum(y2[i], y2[index[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = w * h / area[index[:last]]

        index = np.delete(index, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick]
