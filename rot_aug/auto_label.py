import cv2
import os




def get_w_and_h(img_set):
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

    return w/(w+left+right) , h/(h+top+bottom)





def txt_maker(file_name):
    file_dir = f"./dataset/{file_name}"
    img = cv2.imread(file_dir)
    
    w, h = get_w_and_h(img)

    obj_names = {"a":0, "b":1, "j":2, "g":3, "s":4, "c": 5}
 
    txt_data = ""
    for i in range(5):
        for j in range(5):
            txt_line = f"{obj_names[file_name[0]]} {round(0.1+j*0.2, 6)} {round(0.1+i*0.2, 6)} {w/5} {h/5}\n"
            txt_data += txt_line
    
    save_file_name = file_name.split(".")[0]
    save_dir = f"txt_result/{save_file_name}.txt"
    
    f = open(save_dir, "w")
    f.write(txt_data)
    f.close()
    
    



file_list = os.listdir('dataset')

for idx, file_name in enumerate(file_list):
    txt_maker(file_name)

