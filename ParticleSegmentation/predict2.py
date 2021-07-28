# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:52:26 2019

@author: David
"""
from nets.pspnet import mobilenet_pspnet
import numpy as np
import random
import copy
import os
from PIL import Image

#class_colors = [[0,0,0],[0,255,0]]
NCLASSES = 2
HEIGHT = 256
WIDTH = 256


model = mobilenet_pspnet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
model.load_weights("./logs/ep120-loss0.005-val_loss0.071.h5")
imgs = os.listdir("./img/")
#imgs = os.listdir("C:/Users/David/Workspaces/liushuhui_1_david_lw-Semantic-Segmentation-master/Semantic-Segmentation/pspnet_Mobile/dataset2/image")
for png in imgs:

    img = Image.open("./img/"+png)
    #img = Image.open(("C:/Users/David/Workspaces/liushuhui_1_david_lw-Semantic-Segmentation-master/Semantic-Segmentation/pspnet_Mobile/dataset2/image/"+jpg)
    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img = img.resize((WIDTH,HEIGHT))
    img = np.array(img)
    img = img/255.
    img = img.reshape(-1,HEIGHT,WIDTH,1)
    pr = (model.predict(img)[1])[0]

    pr = pr.reshape((int(HEIGHT), int(WIDTH),NCLASSES)).argmax(axis=-1)

    
#    
#    seg_img = np.zeros((int(HEIGHT/4), int(WIDTH/4),3))
#    colors = class_colors
#
#    for c in range(NCLASSES):
#        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
#        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
#        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
#
    seg_img = Image.fromarray(np.uint8(pr*255.)).resize((orininal_w,orininal_h))

    #image = Image.blend(old_img,seg_img,0.3)
    image = seg_img
    image.save("./img_out/"+png)

