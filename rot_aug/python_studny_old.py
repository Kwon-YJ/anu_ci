import cv2
import numpy as np
import os
import random
import platform

classes = ["apis", "black", "jagnus", "ggoma", "simil", "crabro"]

def img_pic():
    ran_img= []
    
    file_list = os.listdir('croped_test_')

    random.shuffle(file_list)
  
    for i in range(len(file_list)):
        ran_img.append(file_list[1])
        
    print(ran_img[1])
    return ran_img


def padding(img_set):
    h,w,c = img_set.shape

    k = (h**2 + w**2)**(1/2)

    delta_w = k - w
    delta_h = k - h

    top, bottom = int((k - h)//2), int(delta_h - (k - h)//2)
    left, right = int((k - w)//2), int(delta_w - (k - w)//2)

    print(w,h)
    print(int(delta_w), int(delta_h))
    print(top, bottom, left, right)



    new_img = cv2.copyMakeBorder(img_set, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    new_h,new_w,c = new_img.shape

    print(w,h)

    print(f"0.1 0.1 {w/new_w/5} {h/new_h/5}")

    return new_img


def img_rot (img, degree):
    h, w = img.shape[:-1]

    crossLine = int(((w * h + h * w) ** 0.5))
    centerRotatePT = int(w / 2), int(h / 2)
    new_h, new_w = h, w

    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h))

    return result



def img_add(targets):
    BG_img = np.zeros((2400,2400, 3), np.uint8)
    img_cnt = 0

    for i in range(5):

        for j in range(5):
            hpos = 480 * i
            vpos = 480 * j

            target_img = cv2.imread(f'croped_test_/{targets[img_cnt]}')

            #if w > h:
            #    outputs_txt = [[bbox[6], bbox[4], (bbox[0]+bbox[2])/2/640, (bbox[1]+bbox[3])/2/640*w/h, (bbox[2]-bbox[0])/640, (bbox[3]-bbox[1])/640*w/h] for bbox in outputs[0]]
            #else:
            #    outputs_txt = [[bbox[6], bbox[4], (bbox[0]+bbox[2])/2/640*h/w, (bbox[1]+bbox[3])/2, (bbox[2]-bbox[0])/640*h/w, (bbox[3]-bbox[1])] for bbox in outputs[0]]
            
            #f = open(f'{save_file_name[:-4]}.txt', 'w')
            #f.write(save_txt)
            #f.close()

            img_cnt += 1

            target_img = padding(target_img)
            target_img = cv2.resize(target_img, (480, 480))
            # target_img = img_rot(target_img, 14.4*(j*i))

            target_img = img_rot(target_img, (5*j+i)*14.4)
            

            rows, cols, channels = target_img.shape
            roi = BG_img[vpos:rows+vpos, hpos:cols+hpos]

            img2gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            fg = cv2.bitwise_and(target_img, target_img, mask=mask)
            dst = cv2.add(bg, fg)
            BG_img[vpos:rows+vpos, hpos:cols+hpos] = dst

            
    if platform.system() == 'Linux':
        cv2.imwrite(f'{os.getcwd()}/tile_test_2/{random.randrange(0,1000000)}.jpg', BG_img)
    elif platform.system() == 'Windows':
        cv2.imwrite(f'{os.getcwd()}\\tile_test_2\\{random.randrange(0,1000000)}.jpg', BG_img)


# for i in range(5):

img_add(img_pic())




