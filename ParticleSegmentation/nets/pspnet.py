from keras.models import *
from keras.layers import *
from nets.mobilenet import get_mobilenet_encoder
import numpy as np
import keras.layers as KL
from matplotlib import pyplot as plt
import tensorflow as tf
#from keras_segmentation.models.model_utils import get_segmentation_model
#from keras import backend as K


IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1


def resize_image( inp ,  s , data_format ):
	import tensorflow as tf

	return Lambda( 
		lambda x: tf.image.resize_images(
			x , ( K.int_shape(x)[1]*s[0] ,K.int_shape(x)[2]*s[1] ))  
		)( inp )

def pool_block( feats , pool_factor ):


	if IMAGE_ORDERING == 'channels_first':
		h = K.int_shape( feats )[2]
		w = K.int_shape( feats )[3]
	elif IMAGE_ORDERING == 'channels_last':
		h = K.int_shape( feats )[1]
		w = K.int_shape( feats )[2]

	# strides = [16,16],[8,8],[4,4],[2,2]
	pool_size = strides = [int(np.round( float(h) /  pool_factor)), int(np.round(  float(w )/  pool_factor))]
 
	# 进行不同程度的平均
	x = AveragePooling2D(pool_size , data_format=IMAGE_ORDERING , strides=strides, padding='same')( feats )
	
	# 进行卷积
	x = Conv2D(512, (1 ,1 ), data_format=IMAGE_ORDERING , padding='same' , use_bias=False )( x )
	x = BatchNormalization()(x)
	x = Activation('relu' )(x)

	x = resize_image( x , strides , data_format=IMAGE_ORDERING ) 

	return x


def _pspnet( n_classes , encoder ,  input_height=256, input_width=256  ):



#只有最后一层用pspblock
	img_input , levels = encoder( input_height=input_height,input_width=input_width)
	[f1 , f2 , f3 , f4 , grad] = levels
	grad = ( Conv2D(n_classes, (1, 1), padding='same', data_format=IMAGE_ORDERING))(grad)
	EdgeInf = EdgeAttenNet(f1,f2,f3,f4)
	EdgeInf1 = ( Conv2D(n_classes, (1, 1), padding='same', data_format=IMAGE_ORDERING))(EdgeInf)
	EdgeInf2 = Reshape((-1,n_classes))(EdgeInf1) 
	EdgeResult = Softmax(name='EdgeResult' )(EdgeInf2)

	EdgeResult2 = Reshape((input_height,input_width,n_classes))(EdgeResult)
# 	m1=EdgeResult2[0].eval()
# 	plt.imshow(m1.argmax(axis=-1))
# 	plt.show()
# 	EdgeResult3 = Concatenate( axis=MERGE_AXIS)([EdgeResult2,grad])
# 	EdgeResult3 = Conv2D( n_classes,(1,1),data_format=IMAGE_ORDERING, padding='same' )(EdgeResult3)
	
	o = f4

#	o = ( Conv2D(512, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o)
#	o = ( BatchNormalization())(o)
#	o = Activation('relu' )(o)
#	o = resize_image(o,(2,2),data_format=IMAGE_ORDERING)
#
#	o = Concatenate( axis=MERGE_AXIS)([o,f4])
#	o = ( Conv2D(512, (1, 1), padding='same', data_format=IMAGE_ORDERING))(o)
#	o = ( BatchNormalization())(o)


	o = ( Conv2D(256, (3, 3), padding='same', data_format=IMAGE_ORDERING))(f4)
	o = ( BatchNormalization())(o)
	o = Activation('relu' )(o)
	o = resize_image(o,(2,2),data_format=IMAGE_ORDERING)

	o = Concatenate( axis=MERGE_AXIS)([o,f3])
	o = ( Conv2D(256, (1, 1), padding='same', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	#o = Activation('relu' )(o)
 
	o = ( Conv2D(128, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	o = Activation('relu' )(o)
	o = resize_image(o,(2,2),data_format=IMAGE_ORDERING)

	o = Concatenate( axis=MERGE_AXIS)([o,f2])
	o = ( Conv2D(128, (1, 1), padding='same', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	#o = Activation('relu' )(o)
 	
	o = ( Conv2D(64, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	o = Activation('relu' )(o)
	o = resize_image(o,(2,2),data_format=IMAGE_ORDERING)

	o = Concatenate( axis=MERGE_AXIS)([o,f1])
	o = ( Conv2D(64, (1, 1), padding='same', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	
#	o = Concatenate( axis=MERGE_AXIS)([o,EdgeInf])
#	o = ( Conv2D(64, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o)
#	o = ( BatchNormalization())(o)
#	o = Activation('relu' )(o)
	 
	o = Conv2D( n_classes,(3,3),data_format=IMAGE_ORDERING, padding='same' )(o)
	o = Concatenate( axis=MERGE_AXIS)([o,EdgeInf1])
	o = Conv2D( n_classes,(1,1),data_format=IMAGE_ORDERING, padding='same' )(o)
	o = Concatenate( axis=MERGE_AXIS)([o,grad])
	o = Conv2D( n_classes,(1,1),data_format=IMAGE_ORDERING, padding='same' )(o)
	o = Concatenate( axis=MERGE_AXIS)([o,EdgeResult2])
	o = Conv2D( n_classes,(1,1),data_format=IMAGE_ORDERING, padding='same' )(o)
    
	o = Conv2D( n_classes,(3,3),data_format=IMAGE_ORDERING, padding='same' )(o)
	o = Reshape((-1,n_classes))(o)
	predictRegion = Softmax(name='predictRegion')(o)
	
	 
	model = Model([img_input],[EdgeResult,predictRegion])
	return model

def Atten(f):
	return(f[0]*f[1]+f[1])

def EdgeAttenNet( f1,f2,f3,f4 ):
	o2 = resize_image(f2,(2,2),data_format=IMAGE_ORDERING)
	o2 = ( Conv2D(64, (1, 1), padding='same', data_format=IMAGE_ORDERING))(o2)
	o2 = ( Conv2D(64, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o2)
	o2 = ( BatchNormalization())(o2)
	o2 = Activation('relu' )(o2)
	f12 = ( Conv2D(64, (1, 1), padding='same', data_format=IMAGE_ORDERING))(f1)
	f12 = ( Conv2D(64, (3, 3), padding='same', data_format=IMAGE_ORDERING))(f12)
	f12 = ( BatchNormalization())(f12)
	f12 = Activation('relu' )(f12)
	o3 = Concatenate( axis=MERGE_AXIS)([f12,o2])
	o31 = ( Conv2D(64, (1, 1), padding='same', data_format=IMAGE_ORDERING))(o3)
	o3 = ( BatchNormalization())(o31)
	o3 = Activation('sigmoid' )(o3)
	#o3 = Lambda(lambda f1, o3 ,f12: (f1*o3 + f12) )(f1,o3,f12)
	#o3 = Lambda(Atten)([f12,o3])
	mul = Multiply()([o31,o3])
	o3 = Add()([mul,o31])
	o3 = ( Conv2D(128, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o3)
	o3 = ( BatchNormalization())(o3)
	o3 = Activation('relu' )(o3)
# 	f31 = resize_image(f3,(2,2),data_format=IMAGE_ORDERING)
# 	f31 = ( Conv2D(128, (3, 3), padding='same', data_format=IMAGE_ORDERING))(f31)
# # 	f31 = ( BatchNormalization())(f31)
# # 	f31 = Activation('relu' )(f31)
# 	f31 = resize_image(f31,(2,2),data_format=IMAGE_ORDERING)
# 	f31 = ( Conv2D(128, (3, 3), padding='same', data_format=IMAGE_ORDERING))(f31)
# # 	f31 = ( BatchNormalization())(f31)
# # 	f31 = Activation('relu' )(f31)
# 	o4 = Concatenate( axis=MERGE_AXIS)([f31,o3,f12,o2])
# 	o4 = ( Conv2D(128, (1, 1), padding='same', data_format=IMAGE_ORDERING))(o4)
# 	o4 = ( BatchNormalization())(o4)
#  	#o4 = Activation('sigmoid' )(o4)
#  	#o4 = Lambda(Atten)([o3,o4])
# 	mul1 = Multiply()([o3,o4])
# 	o4 = Add()([mul1,o3])
# 	o4 = ( Conv2D(256, (1, 1), padding='same', data_format=IMAGE_ORDERING))(o4)
# 	o4 = Activation('relu' )(o4)
# 	o4 = ( Conv2D(256, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o4)
# 	o4 = ( BatchNormalization())(o4)
# 	o4 = Activation('relu' )(o4)
	return o3

def mobilenet_pspnet( n_classes ,  input_height=256, input_width=256 ):

	model =  _pspnet( n_classes , get_mobilenet_encoder ,  input_height=input_height, input_width=input_width  )
	model.model_name = "mobilenet_pspnet"
	return model
