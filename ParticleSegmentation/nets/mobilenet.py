
from keras.models import *
from keras.layers import *
import keras.backend as K
import keras
from gradient import gradient_mag

IMAGE_ORDERING = 'channels_last'


def relu6(x):
	return K.relu(x, max_value=6)




def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):

	channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
	filters = int(filters * alpha)
	x = ZeroPadding2D(padding=(1, 1), name='conv1_pad', data_format=IMAGE_ORDERING  )(inputs)
	x = Conv2D(filters, kernel , data_format=IMAGE_ORDERING  ,
										padding='valid',
										use_bias=False,
										strides=strides,
										name='conv1')(x)
	x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
	return Activation(relu6, name='conv1_relu')(x)




def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
													depth_multiplier=1, strides=(1, 1), block_id=1):

	channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
	pointwise_conv_filters = int(pointwise_conv_filters * alpha)

	x = ZeroPadding2D((1, 1) , data_format=IMAGE_ORDERING , name='conv_pad_%d' % block_id)(inputs)
	x = DepthwiseConv2D((3, 3) , data_format=IMAGE_ORDERING ,
														 padding='valid',
														 depth_multiplier=depth_multiplier,
														 strides=strides,
														 use_bias=False,
														 name='conv_dw_%d' % block_id)(x)
	x = BatchNormalization(
			axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
	x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

	x = Conv2D(pointwise_conv_filters, (1, 1), data_format=IMAGE_ORDERING ,
										padding='same',
										use_bias=False,
										strides=(1, 1),
										name='conv_pw_%d' % block_id)(x)
	x = BatchNormalization(axis=channel_axis,
																name='conv_pw_%d_bn' % block_id)(x)
	return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)





def get_mobilenet_encoder( input_height=256 ,  input_width=256 , pretrained='imagenet' ):

	alpha=1.0
	depth_multiplier=1
	dropout=1e-3
	n_classes = 2

	img_input = Input(batch_shape=None,shape=(input_height,input_width , 1))
	
	grad = Lambda(gradient_mag)(img_input)
	#grad1 = edge_cal(img_input[:,:,0])
	#grad = Lambda(edge_cal)(img_input)
	
	#grad2 = edge_cal(img_input[1,:,:,0])
	#grad.append(grad1)
	#grad.append(grad2)
	x = _conv_block(img_input, 64, alpha, strides=(1, 1))
	x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1) 
	x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=2)
	f1 = x

	x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,strides=(2, 2), block_id=3)  
	x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=4)
	x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=5) 
	f2 = x

	x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
														strides=(2, 2), block_id=6)  
	x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=7)
	x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=8) 
	f3 = x

	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
														strides=(2, 2), block_id=9) 
	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10) 
	#x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11) 
	#x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=12) 
	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=13) 
	f4 = x 

#	x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
#														strides=(2, 2), block_id=14)  
#	x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=15)
#	x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=16) 
#	f5 = x 

	return img_input , [f1 , f2 , f3 , f4 , grad ]
	#return img_input , grad, [f1 , f2 , f3 , f4]


