import nms
import numpy as np
import cv2

images = [
    ('images/audrey.jpg', np.array([
        (12, 84, 140, 212),
        (24, 84, 152, 212),
        (36, 84, 164, 212),
        (12, 96, 152, 224),
        (24, 108, 152, 236)])),
    ('images/bksomels.jpg', np.array([
        (114, 60, 178, 124),
        (120, 60, 184, 124),
        (114, 66, 178, 130)])),
    ('images/gpripe.jpg', np.array([
        (12, 30, 76, 94),
        (12, 36, 76, 100),
        (72, 36, 200, 164),
        (84, 48, 212, 176)]))]

for (imagePath, boundingBoxes) in images:
    image = cv2.imread(imagePath)
    orig = image.copy()

    for (startX, startY, endX, endY) in boundingBoxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

    pick = nms.non_max_suppression_slow(boundingBoxes)
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Original", orig)
    cv2.imshow("After NMS", image)
    cv2.waitKey(0)