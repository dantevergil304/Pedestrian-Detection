import cv2


def pyramid(image, scale=1.5, min_size=(30, 30)):
    yield image

    while True:
        image = cv2.resize(image, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image


def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[1], step_size):
        for x in range(0, image.shape[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


if __name__ == "__main__":
    image = cv2.imread('images/gpripe.jpg')
    for (i, layer) in enumerate(pyramid(image, 2)):
        cv2.imshow('Layer {}'.format(i + 1), layer)
    cv2.waitKey()
