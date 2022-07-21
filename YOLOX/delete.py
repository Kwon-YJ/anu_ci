#-*- coding:utf-8 -*-
from webbrowser import BackgroundBrowser
import cv2
import numpy as np
import os
import random
import platform

import copy

def get_target_img_name():
    result = []
    file_list = os.listdir('croped_test_')

    apis_list = [img for img in file_list if 'apis' in img]
    black_list = [img for img in file_list if 'black' in img]
    crabro_list = [img for img in file_list if 'crabro' in img]
    ggoma_list = [img for img in file_list if 'ggoma' in img]
    jangsu_list = [img for img in file_list if 'apis' in img]
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


def vertical_overlay(targets, BG_img, BG_tag, vertical_value):
    hpos, vpos = 0, 0
    for idx, img in enumerate(targets):
        target_img_ = cv2.imread(f'croped_test_/{img}')
        target_img = copy.deepcopy(target_img_)

        rows, cols, channels = target_img.shape
        if rows > cols:
            target_img = cv2.resize(target_img, (86, int(86*rows/cols)))
        elif cols > rows:
            target_img = cv2.resize(target_img, (int(152*cols/rows), 152))
        rows, cols, channels = target_img.shape

        if vpos + rows > 2160:
            break
        hpos = int(vertical_value - cols * 0.5)

        roi = BG_img[vpos:rows+vpos, hpos:cols+hpos]
        img2gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(target_img, target_img, mask=mask)
        dst = cv2.add(bg, fg)
        BG_img[vpos:rows+vpos, hpos:cols+hpos] = dst

        font = cv2.FONT_HERSHEY_SIMPLEX
        #text = img.split('_')[0]
        #classes = ['apis', 'black', 'crabro', 'simil', 'jangsu', 'ggoma']
        #class_id = classes.index(text)
        #cntrX = vertical_value
        #cntrY = vpos + (0.5*rows)
        # txt_data += f'{class_id} {cntrX} {cntrY} {rows} {cols}\n'
        #txt_data[0] = txt_data[0] + f'{class_id} {cntrX} {cntrY} {rows} {cols}\n'

        continue
        txt_size = cv2.getTextSize(text, font, 1, 1)[0]
        cv2.putText(target_img, text, (0, txt_size[1]), font, 1,(255,255,255), thickness=2)

        roi = BG_tag[vpos:rows+vpos, hpos:cols+hpos]
        img2gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(target_img, target_img, mask=mask)
        dst = cv2.add(bg, fg)

        BG_tag[vpos:rows+vpos, hpos:cols+hpos] = dst
        vpos += rows + 5

    return BG_img, BG_tag , idx


def horizontal_overlay(targets, BG_img, BG_tag, horizontal_value):
    hpos, vpos = 0, 0
    for idx, img in enumerate(targets):
        target_img_ = cv2.imread(f'croped_test_/{img}')

        target_img = copy.deepcopy(target_img_)

        rows, cols, channels = target_img.shape
        if rows > cols:
            target_img = cv2.resize(target_img, (86, int(86*rows/cols)))
        elif cols > rows:
            target_img = cv2.resize(target_img, (int(152*cols/rows), 152))
        rows, cols, channels = target_img.shape

        if hpos + rows > 3840:
            break
        if hpos+rows>1280-175 and hpos+rows<1280+175 or hpos+rows>2560-175 and hpos+rows<2560+175:
            hpos += cols + 5
            continue
        

        vpos = int(horizontal_value - rows * 0.5)
        roi = BG_img[vpos:rows+vpos, hpos:cols+hpos]
        img2gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        if roi.shape[1] != mask.shape[1]:
            continue
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(target_img, target_img, mask=mask)
        dst = cv2.add(bg, fg)
        BG_img[vpos:rows+vpos, hpos:cols+hpos] = dst


        font = cv2.FONT_HERSHEY_SIMPLEX
        text = img.split('_')[0]


        continue
        txt_size = cv2.getTextSize(text, font, 1, 1)[0]
        cv2.putText(target_img, text, (0, txt_size[1]), font, 1,(255,255,255), thickness=2)

        roi = BG_tag[vpos:rows+vpos, hpos:cols+hpos]
        img2gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        if roi.shape[1] != mask.shape[1]:
            continue
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(target_img, target_img, mask=mask)
        dst = cv2.add(bg, fg)
        BG_tag[vpos:rows+vpos, hpos:cols+hpos] = dst

        
        hpos += cols + 5
    return BG_img, BG_tag, idx

def img_add(targets):

    while 1:
        file_name = random.randrange(0,1000000)
        if file_name > 100000:
            break

    BG_img = np.zeros((2160,3840, 3), np.uint8)
    BG_tag = np.zeros((2160,3840, 3), np.uint8)

    BG_img, BG_tag, idx = vertical_overlay(targets, BG_img, BG_tag, 1280)
    BG_img, BG_tag, idx = vertical_overlay(targets[idx:], BG_img, BG_tag, 2560)
    targets = targets[::-1]

    BG_img, BG_tag, idx = horizontal_overlay(targets, BG_img, BG_tag, 720)
    BG_img, BG_tag, idx = horizontal_overlay(targets[idx:], BG_img, BG_tag, 1440)

    #cv2.imshow('result', BG_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    if platform.system() == 'Linux':
        cv2.imwrite(f'{os.getcwd()}/tile_test_/{file_name}.jpg', BG_img)
    elif platform.system() == 'Windows':
        cv2.imwrite(f'{os.getcwd()}\\tile_test_\\{file_name}.jpg', BG_img)
    
    return file_name

    #if platform.system() == 'Linux':
    #    cv2.imwrite(f'{os.getcwd()}/tile_test_/a{file_name}_tag.jpg', BG_tag)
    #elif platform.system() == 'Windows':
    #    cv2.imwrite(f'{os.getcwd()}\\tile_test_\\a{file_name}_tag.jpg', BG_tag)


global txt_data
txt_data = ['']

target_list = get_target_img_name()
file_name = img_add(target_list)

f = open(f'{file_name}.txt', 'w')
f.write(txt_data[0])
f.close()
txt_data = ''


exit()
for i in range(150):
    target_list = get_target_img_name()
    img_add(target_list)



