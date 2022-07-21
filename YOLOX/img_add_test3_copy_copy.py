#-*- coding:utf-8 -*-
from cgi import test
from webbrowser import BackgroundBrowser
import cv2
import numpy as np
import os
import random
import platform


def get_target_img_name():
    result = []
    file_list = os.listdir('croped_test_')

    apis_list = [img for img in file_list if 'apis' in img]
    black_list = [img for img in file_list if 'black' in img]
    crabro_list = [img for img in file_list if 'crabro' in img]
    ggoma_list = [img for img in file_list if 'ggoma' in img]
    jangsu_list = [img for img in file_list if 'jangsu' in img]
    simil_list = [img for img in file_list if 'simil' in img]

    random.shuffle(apis_list)
    random.shuffle(black_list)
    random.shuffle(crabro_list)
    random.shuffle(ggoma_list)
    random.shuffle(jangsu_list)
    random.shuffle(simil_list)
    
    for i in range(len(apis_list)):
        result.append(apis_list[i])
        result.append(black_list[i])
        result.append(crabro_list[i])
        result.append(ggoma_list[i])
        result.append(jangsu_list[i])
        result.append(simil_list[i])
    return result



def padding(img, set_size):
    h,w,c = img.shape
    delta_w = set_size - w
    delta_h = set_size - h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    new_h, new_w, new_c = new_img.shape
    # print(h,w,'  ',new_h,new_w)
    
    ratio = [h/w, w/h]
    
    return new_img, ratio


def img_add(targets):
    txt_data = ''
    BG_img = np.zeros((2160,3840, 3), np.uint8)
    img_cnt = 0

    for i in range(16):
        for j in range(9):
            hpos = 240 * i
            vpos = 240 * j

            target_img = cv2.imread(f'croped_test_/{targets[img_cnt]}')
            img_cnt += 1

            target_img, ratio = padding(target_img, max(target_img.shape))


            classes = ['apis', 'black', 'crabro', 'simil', 'jangsu', 'ggoma']

            text = targets[img_cnt].split('_')[0]

            class_id = classes.index(text)


            cntrX = (hpos + 120) / 3840
            cntrY = (vpos + 120) / 2160

            if ratio[0] > ratio[1]:
                width = ratio[1] * 240 / 3840
                height = 240 / 2160
            else:
                width = 240 / 3840
                height = ratio[0] * 240 / 2160
            
            txt_data = txt_data + f'{class_id} {cntrX} {cntrY} {width} {height}\n'
            



            target_img = cv2.resize(target_img, (240, 240))

            rows, cols, channels = target_img.shape
            roi = BG_img[vpos:rows+vpos, hpos:cols+hpos]

            img2gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            fg = cv2.bitwise_and(target_img, target_img, mask=mask)
            dst = cv2.add(bg, fg)
            BG_img[vpos:rows+vpos, hpos:cols+hpos] = dst

    file_name = random.randrange(0,1000000)

    if platform.system() == 'Linux':
        cv2.imwrite(f'{os.getcwd()}/tile_test_5/{file_name}.jpg', BG_img)
    elif platform.system() == 'Windows':
        cv2.imwrite(f'{os.getcwd()}\\tile_test_5\\{file_name}.jpg', BG_img)

    f = open(f'./tile_test_5/{file_name}.txt', 'w')
    f.write(txt_data)
    f.close()




for i in range(150):
    img_add(get_target_img_name())



