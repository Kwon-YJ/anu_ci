from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy





def use_resnet(img):
    weights = ResNet50_Weights.DEFAULT
    model = torch.load("temp_resnet.pth")
    model.eval()
    img = img.to("cuda:0")
    preprocess = weights.transforms()
    batch = preprocess(img).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = ["apis", "black", "jangsu", "ggoma", "crabro", "simil"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")







# img = read_image('C:\\Users\\CILAB\\Documents\\github\\anu_ci\\resnet\\data\\vespa\\train\\simil\\simil0802.jpg')


img = read_image("C:\\Users\\CILAB\\YOLOX\\test_\\apis_t_000.jpg")


print(img.shape)

exit()
use_resnet(img)







'''
img = read_image('C:\\Users\\CILAB\\Documents\\github\\anu_ci\\resnet\\data\\vespa\\train\\simil\\simil0802.jpg')


# device = torch.device("cuda")
weights = ResNet50_Weights.DEFAULT
model = torch.load("temp_resnet.pth")
model.eval()
img = img.to("cuda:0")
preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0)
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
category_name = ["apis", "black", "jangsu", "ggoma", "crabro", "simil"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
'''







'''
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
'''







