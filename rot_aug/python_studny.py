import cv2
import numpy as np
import os
import random
import platform

import math


def img_pick():
    ran_img= []
    
    file_list = os.listdir('croped_test_')

    #random.shuffle(file_list)
  
    #for i in range(len(file_list)):
    #    ran_img.append(file_list[1])

    return file_list


def after_rotate_w_and_h(x, y, theta):
    radian = theta * (math.pi / 180.0)
    x, y = int(0.5*x), int(0.5*y) # 이미지의 중심점이 회전 축

    new_x = x*math.cos(radian) - y*math.sin(radian)
    new_y = x*math.sin(radian) + y*math.cos(radian)
    height = new_y * 2
    #print(f"x y : {x} {y}")
    print(f"after_rotate_w_and_h 1 : new x = {new_x}, new y = {new_y}")
    y = -y
    new_x = x*math.cos(radian) - y*math.sin(radian)
    new_y = x*math.sin(radian) + y*math.cos(radian)
    print(f"after_rotate_w_and_h 2 : new x = {new_x}, new y = {new_y}")
    width = new_x * 2
    return height, width


def padding(img_set):
    h,w,c = img_set.shape


    '''
    new_w = 0
    new_h = 0
    if w % 2 == 0:
        new_w = w-1
    if h % 2 == 1:
        new_h = h-1
    img_set = img_set[0:new_h,0:new_w].copy()
    h,w,c = img_set.shape
    '''

    k = (h**2 + w**2)**(1/2)

    delta_w = k - w
    delta_h = k - h

    top, bottom = int((k - h)//2), int(delta_h - (k - h)//2)
    left, right = int((k - w)//2), int(delta_w - (k - w)//2)

    print(f"original : {w},{h}")

    if h % 2 == 0:
        top, bottom = max(top, bottom), max(top, bottom)
    if w % 2 == 0:
        left, right = max(left, right), max(left, right)

    print(top, bottom, left, right)

    print(h + top + bottom)

    print(left + right+ w)

    new_img = cv2.copyMakeBorder(img_set, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


    return new_img




def img_rot (img, degree):
    h, w = img.shape[:-1]

    crossLine = int(((w * h + h * w) ** 0.5))
    centerRotatePT = int(w / 2), int(h / 2)
    new_h, new_w = h, w

    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h))

    return result

import copy


def img_add(idx, targets):
    
    original_img = cv2.imread(f'dataset/{targets}')

    pix = original_img[2,2]
    
    BG_img = np.full((2400,2400, 3), pix, np.uint8)
    
    for i in range(5):

        for j in range(5):
            hpos = 480 * i
            vpos = 480 * j

            target_img = copy.deepcopy(original_img)



            target_img = padding(target_img)
            target_img = cv2.resize(target_img, (480, 480))

            target_img = img_rot(target_img, (5*j+i)*14.5)


            #cv2.imshow("temp", target_img)
            #cv2.waitKey()
            #cv2.destroyAllWindows()


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
        cv2.imwrite(f'{os.getcwd()}/tile_test_2/sample_b/{targets}.jpg', BG_img)
    elif platform.system() == 'Windows':
        cv2.imwrite(f'{os.getcwd()}\\tile_test_2\\sample_b\\{targets}.jpg', BG_img)

file_list = os.listdir('dataset')

print(file_list)

for idx, file_name in enumerate(file_list):
    img_add(idx, file_name)



