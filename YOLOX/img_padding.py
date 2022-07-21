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



def padding(img, set_size):
    h,w,c = img.shape
    delta_w = set_size - w
    delta_h = set_size - h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return new_img

def img_add(target):
    target_img = cv2.imread(f'croped_test_/{target}')
    target_img = padding(target_img, max(target_img.shape))
    target_img = cv2.resize(target_img, (240, 240))

    if platform.system() == 'Linux':
        cv2.imwrite(f'{os.getcwd()}/croped_test_padding/{target}', target_img)
    elif platform.system() == 'Windows':
        cv2.imwrite(f'{os.getcwd()}\\croped_test_padding\\{target}', target_img)


targets = os.listdir('croped_test_')


for target in targets:
    img_add(target)


