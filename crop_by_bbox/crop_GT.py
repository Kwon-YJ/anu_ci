#-*- coding:utf-8 -*-
import cv2
import os
import platform


def get_target_img_name_list(data_dir):
    result = []
    file_list = os.listdir(data_dir)
    return [img_name[:-4] for img_name in file_list if 'txt' not in img_name]

def get_jpg_and_txt(folder_name, file_name):
    if platform.system() == 'Linux':
        target_img = cv2.imread(f'{folder_name}/{file_name}.jpg')
        with open(f'{folder_name}/{file_name}.txt', 'r') as file:
            target_txt = file.readlines()
    elif platform.system() == 'Windows':
        target_img = cv2.imread(f'{folder_name}\\{file_name}.jpg')
        with open(f'{folder_name}\\{file_name}.txt', 'r') as file:
            target_txt = file.readlines()
    return target_img, target_txt

folder_name = 'original_dataset'
target_list = get_target_img_name_list(folder_name)

for file_name in target_list:
    target_img, target_txt = get_jpg_and_txt(folder_name, file_name)
    rows, cols, _ = target_img.shape

    for idx, bbox_data in enumerate(target_txt):
        bbox_data = bbox_data.split(" ")
        bbox_data = list(map(float, bbox_data))

        x1 = (cols * (bbox_data[1] - 0.5 * bbox_data[3])) * 0.98
        x2 = (cols * (bbox_data[1] + 0.5 * bbox_data[3])) * 1.02
        y1 = (rows * (bbox_data[2] - 0.5 * bbox_data[4])) * 0.98
        y2 = (rows * (bbox_data[2] + 0.5 * bbox_data[4])) * 1.02

        result = target_img[int(y1):int(y2),int(x1):int(x2)].copy()

        if platform.system() == 'Linux':
            if idx != 0:
                save_name = f"./result/{file_name}_idx.jpg"
            else:
                save_name = f"./result/{file_name}.jpg"
        elif platform.system() == 'Windows':
            if idx != 0:
                save_name = f".\\result\\{file_name}_{idx}.jpg"
            else:
                save_name = f".\\result\\{file_name}.jpg"

        cv2.imwrite(save_name, result)

        #cv2.imshow('result', result)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


