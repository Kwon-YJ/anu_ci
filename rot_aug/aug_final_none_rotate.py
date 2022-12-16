import cv2
import numpy as np
import os
import random
import platform
import copy


def padding(img_set):
    h,w,c = img_set.shape

    k = (h**2 + w**2)**(1/2)

    delta_w = k - w
    delta_h = k - h

    top, bottom = int((k - h)//2), int(delta_h - (k - h)//2)
    left, right = int((k - w)//2), int(delta_w - (k - w)//2)

    if h % 2 == 0:
        top, bottom = max(top, bottom), max(top, bottom)
    if w % 2 == 0:
        left, right = max(left, right), max(left, right)

    new_img = cv2.copyMakeBorder(img_set, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img


def img_rot (img, degree):
    degree = 0
    h, w = img.shape[:-1]

    crossLine = int(((w * h + h * w) ** 0.5))
    centerRotatePT = int(w / 2), int(h / 2)
    new_h, new_w = h, w

    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h))

    return result


def get_BG(targets):
    
    original_img = cv2.imread(f'dataset/{targets}', cv2.IMREAD_UNCHANGED)
    
    pix = original_img[0:20, 0:20]
    pix = cv2.resize(pix, (240,240))

    BG_img = np.zeros((2400,2400, 3), np.uint8)

    for i in range(10):     # for i  in range (2400 / resize)
        for j in range(10):

            hpos = 240 * i  #hpos =  resize * i
            vpos = 240 * j

            target_img = pix

            rows, cols, channels = target_img.shape
            roi = BG_img[vpos:rows+vpos, hpos:cols+hpos]

            img2gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            fg = cv2.bitwise_and(target_img, target_img, mask=mask)
            dst = cv2.add(bg, fg)
            BG_img[vpos:rows+vpos, hpos:cols+hpos] = dst


    #cv2.imshow("123",BG_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return BG_img



def img_add(idx, targets):
    original_img = cv2.imread(f'dataset/{targets}', cv2.IMREAD_UNCHANGED)


    BG_img = get_BG(targets)
    
    for i in range(5):

        for j in range(5):
            hpos = 480 * i
            vpos = 480 * j

            target_img = copy.deepcopy(original_img)

            target_img = padding(target_img)
            target_img = cv2.resize(target_img, (480, 480))


            target_img = img_rot(target_img, (5*j+i)*14.4)
            

            rows, cols, channels = target_img.shape
            roi = BG_img[vpos:rows+vpos, hpos:cols+hpos]

            img2gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            fg = cv2.bitwise_and(target_img, target_img, mask=mask)
            dst = cv2.add(bg, fg)
            BG_img[vpos:rows+vpos, hpos:cols+hpos] = dst


    if platform.system() == 'Linux':
        cv2.imwrite(f'{os.getcwd()}/result/{targets}', BG_img)
    elif platform.system() == 'Windows':
        cv2.imwrite(f'{os.getcwd()}\\result\\{targets}', BG_img)


file_list = os.listdir('dataset')

print(file_list)

for idx, file_name in enumerate(file_list):
    img_add(idx, file_name)



