# -*- coding: utf-8 -*-


import tensorflow as tf

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
#def edge_cal(image):
#	#GaussianBlur图像高斯平滑处理
#	image1 = image[:,:,:,0]
#	image1 = np.array(image1*255.0, dtype=np.uint8)
#	blurred = cv2.GaussianBlur(image1, (15, 15), 5)
##	plt.imshow(blurred, cmap='gray')
##	plt.show()
## 	dst = cv2.Laplacian(blurred, cv2.CV_64F, ksize = 3) #再通过拉普拉斯算子做边缘检测
## 	edge_output = cv2.convertScaleAbs(dst)
#	edge_output = cv2.Canny(blurred, 50, 200,apertureSize = 5)
#	edge_output = edge_output/255.
#	edge_output=np.expand_dims(edge_output,axis=-1)
#	return edge_output
#	plt.imshow(edge_output,cmap='gray')#输出灰度图像
#	plt.show()
#    return edge_output
	
	
#if __name__ == '__main__':
#   img = cv2.imread(r'C:\Users\Administrator\Desktop\1.png', cv2.IMREAD_UNCHANGED)
#  
#   # cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
#   plt.imshow(img,cmap='gray')
#   plt.show()
#   edge_cal(img)

#   cv2.waitKey(0)#等待键盘输入，不输入 则无限等待
#   cv2.destroyAllWindows()#清除所以窗口 
def _all_close(x, y, rtol=1e-5, atol=1e-8):

    return tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol)

def gradient_mag(tensor, from_rgb=False, eps=1e-12):

    if from_rgb:

        tensor = tf.image.rgb_to_grayscale(tensor[..., :3])

    tensor_edge = tf.image.sobel_edges(tensor)



    def _normalised_mag():

        mag = tf.reduce_sum(tensor_edge ** 2, axis=-1) + eps

        mag = tf.math.sqrt(mag)

        mag /= tf.reduce_max(mag, axis=[1, 2], keepdims=True)

        return mag



    z = tf.zeros_like(tensor)

    normalised_mag = tf.cond(

        _all_close(tensor_edge, tf.zeros_like(tensor_edge)),

        lambda: z,

        _normalised_mag, name='potato')



    return normalised_mag