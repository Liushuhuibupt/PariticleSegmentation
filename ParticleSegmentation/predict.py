# -*- coding: utf-8 -*-

from nets.pspnet import mobilenet_pspnet
import numpy as np
import random
import copy
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt


#class_colors = [[0,0,0],[0,255,0]]
NCLASSES = 2
HEIGHT = 256
WIDTH = 256


model = mobilenet_pspnet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
model.load_weights("./logs/ep100-loss0.004-val_loss0.060.h5")
imgs = os.listdir("./img/")
#imgs = os.listdir("C:/Users/David/Workspaces/liushuhui_1_david_lw-Semantic-Segmentation-master/Semantic-Segmentation/pspnet_Mobile/dataset2/image")
for png in imgs:
    
    img = cv2.imread("./img/"+png,cv2.IMREAD_UNCHANGED)
    #img = cv2.imread("./img/"+png)
    old_img = img.copy()
    #old_img=cv2.resize(old_img,(WIDTH,HEIGHT))
    #img = img[:,:,0]
    # = img.shape[0]
    #orininal_w = img.shape[1]
    #img.resize((WIDTH,HEIGHT))
    #img = np.array(img)
    img = img/255.
    img = img.reshape(-1,HEIGHT,WIDTH,1)
    pr = (model.predict(img)[1])[0]
    #['predictRegion'][0]
    pr = pr.reshape((int(HEIGHT), int(WIDTH),NCLASSES)).argmax(axis=-1)
  
    plt.imshow(pr)
    
    seg_img = (np.uint8(pr*255.))
    #seg_img.resize(orininal_w,orininal_h)
    #seg_img=cv2.resize(seg_img,(orininal_w,orininal_h))
    #color_img = cv2.cvtColor(old_img,cv2.COLOR_Gra2) 
    contours, hierarchy = cv2.findContours(seg_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        cv2.drawContours(old_img, [c], -1, (255, 0, 0), 1)

    cv2.imwrite("./img_out_contour_diceloss/"+png,old_img)
#    img = Image.open("./img/"+png)
#    #img = Image.open(("C:/Users/David/Workspaces/liushuhui_1_david_lw-Semantic-Segmentation-master/Semantic-Segmentation/pspnet_Mobile/dataset2/image/"+jpg)
#    old_img = copy.deepcopy(img)
#    orininal_h = np.array(img).shape[0]
#    orininal_w = np.array(img).shape[1]
#
#    img = img.resize((WIDTH,HEIGHT))
#    img = np.array(img)
#    img = img/255.
#    img = img.reshape(-1,HEIGHT,WIDTH,1)
#    pr = model.predict(img)[0]
#
#    pr = pr.reshape((int(HEIGHT), int(WIDTH),NCLASSES)).argmax(axis=-1)
#
#    
##    
##    seg_img = np.zeros((int(HEIGHT/4), int(WIDTH/4),3))
##    colors = class_colors
##
##    for c in range(NCLASSES):
##        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
##        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
##        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
##
#    seg_img = Image.fromarray(np.uint8(pr*255.)).resize((orininal_w,orininal_h))
#
#    image = Image.blend(old_img,seg_img,0.3)
#    #image = seg_img
#    image.save("./img_out_blend/"+png)

