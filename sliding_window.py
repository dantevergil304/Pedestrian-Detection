from helpers import pyramid
from helpers import sliding_window
import cv2
import time

if __name__ == "__main__":
    image = cv2.imread('images/pedestrian.jpg')
    print(image.shape)
    (win_height, win_width) = (128, 128)
    for layer in pyramid(image):
        for (x, y, window) in sliding_window(layer, step_size=32, window_size=(win_width, win_height)):
            if window.shape[0] != win_height or window.shape[1] != win_width:
                continue
            clone = layer.copy()
            cv2.rectangle(clone, (x, y), (x + win_width, y + win_height), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)
