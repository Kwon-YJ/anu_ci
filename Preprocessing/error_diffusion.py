
import cv2
import numpy as np

import copy

# target_img = "./Leonardo_da_Vinci_(1452-1519)_-_The_Last_Supper_(1495-1498).jpg"
target_img = "./img_file/black1430.jpg"

img = cv2.imread(target_img)

cv2.imshow("original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cv2.imshow("gray", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

height, width = img.shape


zoom_img = np.zeros((int(height*2), int(width*2)))

# zoom_img = np.zeros((int(height), int(width)))

for i in range(height):
    for j in range(width):
        pixel = img[i][j] / 255
        zoom_img[i*2][j*2] = pixel      # (0, 0)
        zoom_img[i*2][j*2+1] = pixel    # (0, 1)
        zoom_img[i*2+1][j*2] = pixel    # (1, 0)
        zoom_img[i*2+1][j*2+1] = pixel  # (1, 1)

cv2.imshow("zoom", zoom_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


