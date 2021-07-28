from nets.pspnet import mobilenet_pspnet
from keras.optimizers import Adam, RMSprop,SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import keras
import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.utils import plot_model
from keras.utils import multi_gpu_model
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
 


NCLASSES = 2
HEIGHT = 256
WIDTH = 256
l1=0.0

# def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
#                     mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
#                     flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (1024,1024),seed = 1):
#     '''
#     can generate image and mask at the same time
#     use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
#     if you want to visualize the results of generator, set save_to_dir = "your path"
#     '''
#     image_datagen = ImageDataGenerator(**aug_dict)
#     mask_datagen = ImageDataGenerator(**aug_dict)
#     image_generator = image_datagen.flow_from_directory(
#         train_path,
#         classes = [image_folder],
#         class_mode = None,
#         color_mode = image_color_mode,
#         target_size = target_size,
#         batch_size = batch_size,
#         save_to_dir = save_to_dir,
#         save_prefix  = image_save_prefix,
#         seed = seed)
#     mask_generator = mask_datagen.flow_from_directory(
#         train_path,
#         classes = [mask_folder],
#         class_mode = None,
#         color_mode = mask_color_mode,
#         target_size = target_size,
#         batch_size = batch_size,
#         save_to_dir = save_to_dir,
#         save_prefix  = mask_save_prefix,
#         seed = seed)
#     train_generator = zip(image_generator, mask_generator)
#     #train_generator = itertools.izip(image_generator, mask_generator)
#     for (img,mask) in train_generator:
#         img,mask = adjustData(img,mask,flag_multi_class,num_class)
#         yield (img,mask)



# def testGenerator(test_path,num_image = 60,target_size = (1024,1024),flag_multi_class = False,as_gray = True):
#     for i in range(num_image):
#         img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
#         img = img / 255
#         img = trans.resize(img,target_size)
#         img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
#         img = np.reshape(img,(1,)+img.shape)
#         yield img

image_datagen = ImageDataGenerator(rotation_range=90,
                    width_shift_range=0.00,
                    height_shift_range=0.00,
                    shear_range=0.00,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='reflect')
edge_datagen = ImageDataGenerator(rotation_range=90,
                    width_shift_range=0.00,
                    height_shift_range=0.00,
                    shear_range=0.00,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='reflect')
mask_datagen = ImageDataGenerator(rotation_range=90,
                    width_shift_range=0.00,
                    height_shift_range=0.00,
                    shear_range=0.00,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='reflect')

def generate_arrays_from_file(lines,batch_size):  
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        Y_edge = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = Image.open(r"./dataset2/image" + '/' + name)
            # img = image_datagen.random_transform(img, seed=1)
            #plt.imshow(img)
            #plt.show()
            img = img.resize((int(WIDTH),int(HEIGHT)))
            img = np.array(img)
            #img = img/255.
            img=np.expand_dims(img,axis=-1)
            img = image_datagen.random_transform(img, seed=1)
            #plt.imshow(img[:,:,0],cmap ='gray')
#            plt.show()
            img=img/255.
            X_train.append(img)
  

            name = lines[i].split(';')[1]
            # 从文件中读取图像
            img = Image.open(r"./dataset2/edge" + '/' + name)
            # img = image_datagen.random_transform(img, seed=1)
            #plt.imshow(img)
            #plt.show()
            img = img.resize((int(WIDTH),int(HEIGHT)))
            img = np.array(img)
            #img = img/255.
            img=np.expand_dims(img,axis=-1)
            img = edge_datagen.random_transform(img, seed=1)
            # plt.imshow(img,cmap ='gray')
            # plt.show()
            img=img/255.
            img.astype(int)
            edge_labels = np.zeros((int(HEIGHT),int(WIDTH),NCLASSES))
            for c in range(NCLASSES):
                edge_labels[: , : , c ] = (img[:,:,0] == c ).astype(int)
            edge_labels = np.reshape(edge_labels, (-1,NCLASSES))
            
            Y_edge.append(edge_labels)




            name = (lines[i].split(';')[2]).replace("\n", "")
            # 从文件中读取图像
            img = Image.open(r"./dataset2/label" + '/' + name)
            # img = mask_datagen.random_transform(img, seed=1)
            #plt.imshow(img)
            #plt.show()
            img = img.resize((int(WIDTH),int(HEIGHT)))
            img = np.array(img)
            # img=img/255.
            
            img=np.expand_dims(img,axis=-1)
            img = mask_datagen.random_transform(img, seed=1)#与image_datagen相同的种子，保证变换是一样的
#            plt.imshow(img[:,:,0],cmap ='gray')
#            plt.show()
            img=img/255.
            img.astype(int)
            #img=np.reshape(img,(int(WIDTH/4),int(HEIGHT/4),1))
            #seg_labels = np.zeros((int(HEIGHT/4),int(WIDTH/4),NCLASSES))
            seg_labels = np.zeros((int(HEIGHT),int(WIDTH),NCLASSES))
            for c in range(NCLASSES):
                seg_labels[: , : , c ] = (img[:,:,0] == c ).astype(int)
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))
            Y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i+1) % n
        yield (np.array(X_train),{'EdgeResult':np.array(Y_edge),'predictRegion':np.array(Y_train)})

def loss(y_true, y_pred):
    loss = K.categorical_crossentropy(y_true,y_pred)
    return loss

def dyn_weighted_bincrossentropy(true, pred):

    """

    Calculates weighted binary cross entropy. The weights are determined dynamically

    by the balance of each category. This weight is calculated for each batch.

    
    The weights are calculted by determining the number of 'pos' and 'neg' classes 

    in the true labels, then dividing by the number of total predictions.


    For example if there is 1 pos class, and 99 neg class, then the weights are 1/100 and 99/100.

    These weights can be applied so false negatives are weighted 99/100, while false postives are weighted

    1/100. This prevents the classifier from labeling everything negative and getting 99% accuracy.


    This can be useful for unbalanced catagories.

    """

    # get the total number of inputs
    true=true[:,:,1]
    pred=pred[:,:,1]#在这里，取第二个维度对应是的前景

    num_pred = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) + keras.backend.sum(true)

    

    # get weight of values in 'pos' category

    zero_weight =  keras.backend.sum(true)/ num_pred +  keras.backend.epsilon() 

    

    # get weight of values in 'false' category

    one_weight = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) / num_pred +  keras.backend.epsilon()



    # calculate the weight vector

    weights =  (1.0 - true) * zero_weight +  true * one_weight 

    

    # calculate the binary cross entropy

    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)

    

    # apply the weights

    weighted_bin_crossentropy = weights * bin_crossentropy 



    return keras.backend.mean(weighted_bin_crossentropy)





def weighted_bincrossentropy(true, pred, weight_zero = 0.25, weight_one = 1):

    """

    Calculates weighted binary cross entropy. The weights are fixed.

        

    This can be useful for unbalanced catagories.

    

    Adjust the weights here depending on what is required.

    

    For example if there are 10x as many positive classes as negative classes,

        if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives 

        will be penalize 10 times as much as false negatives.



    """
    true=true[:,:,1]
    pred=pred[:,:,1]#在这里，取第二个维度对应是的前景
  

    # calculate the binary cross entropy

    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)

    

    # apply the weights

    weights = true * weight_one + (1. - true) * weight_zero

    weighted_bin_crossentropy = weights * bin_crossentropy 



    return keras.backend.mean(weighted_bin_crossentropy)

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true=y_true[:,:,1]
    y_pred=y_pred[:,:,1]#在这里，取第二个维度对应是的前景
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return K.mean( ((2. * intersection + smooth) / (union + smooth)))
def dice_coef_loss(y_true, y_pred):
	return (1 - dice_coef(y_true, y_pred, smooth=1e-6))
	
#def dice_coef_loss1(y_true, y_pred,smooth=1e-6):
#    y_true=y_true[:,:,0]
#    y_pred=y_pred[:,:,0]
#    intersection = K.sum(y_true * y_pred)
#    union = K.sum(y_true) + K.sum(y_pred)
#    return (1-K.mean( ((2. * intersection + smooth) / (union + smooth))))


def focal_loss(y_true, y_pred,gamma=2., alpha=.25):


    y_true=y_true[:,:,1]
    y_pred=y_pred[:,:,1]

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    return (-K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0)))

def tversky(y_true, y_pred):
    y_true=y_true[:,:,1]
    y_pred=y_pred[:,:,1]
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + 1e-6)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + 1e-6)
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)


def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1])
    union = K.sum(y_true, axis=[1]) + K.sum(y_pred, axis=[1]) - intersection
    return  K.mean((intersection + eps) / (union + eps))

def MeanIoU(y_true, y_pred, eps=1e-6):
    one=IoU(y_true[:,:,0],y_pred[:,:,0],eps=1e-6)
    two=IoU(y_true[:,:,1],y_pred[:,:,1],eps=1e-6)
    return (one+two)/2.

def IoUloss(y_true, y_pred):
    return (1-IoU(y_true[:,:,1], y_pred[:,:,1]))

#def lovasz_softmax_loss(y_true, y_pred):
	







if __name__ == "__main__":
    log_dir = "logs/"
    # 获取model
    model = mobilenet_pspnet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
	
    #model = multi_gpu_model(model, gpus=2)
	
    # model.summary()
#    BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
#										'releases/download/v0.6/')
    model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ( '1_0' , 224 )
   
#    weight_path = BASE_WEIGHT_PATH + model_name
#    weights_path = keras.utils.get_file(model_name, weight_path )
#    print(weight_path)
#    model.load_weights(weights_path,by_name=True,skip_mismatch=True)
    
    #plot_model(model, to_file='model.png')
    model.summary()
    # 打开数据集的txt
    with open(r".\dataset2\train.txt","r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 80%用于训练，20%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    # 保存的方式，1世代保存一次
    checkpoint_period = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', 
                                    verbose =1,
                                    save_weights_only=True, 
                                    save_best_only=False, 
                                    period=1
                                )
    # 学习率下降的方式，val_loss三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss', 
                            factor=0.5, 
                            patience=20, 
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=10, 
                            verbose=1
                        )



    model.compile(loss={'EdgeResult':  IoUloss, 'predictRegion': loss},
                 loss_weights = {'EdgeResult': 0.2, 'predictRegion': 0.8},
            optimizer = Adam(lr=1e-4, decay=0.00002),
            #optimizer = SGD(lr=3e-4, momentum=0.9),
            metrics = {'EdgeResult': ['accuracy',MeanIoU,dice_coef], 'predictRegion': ['accuracy',MeanIoU,dice_coef]})
    batch_size = 6
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # 开始训练
    history=model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size), 
            validation_steps=max(1, num_val//batch_size),
            epochs=150,
            initial_epoch=0,
            callbacks=[checkpoint_period, reduce_lr])

    model.save_weights(log_dir+'diceloss_last1.h5')
    
    print(history.history)#打印结果
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss1 = history.history['loss']
    # val_loss = history.history['val_loss']
    # meanIoU = history.history['MeanIoU']
    # val_meanIoU = history.history['val_MeanIoU']
    # epochs = range(1, len(acc) + 1)
    # plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    # plt.grid()
    # plt.savefig('bincrossentropy_accuracy.jpg')
    
    # plt.figure()
    # plt.plot(epochs, loss1, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # plt.grid()
    # plt.savefig('bincrossentropy_loss.jpg')
    # #plt.show()
    
    # plt.figure()
    # plt.plot(epochs, meanIoU, 'bo', label='Training meanIoU')
    # plt.plot(epochs, val_meanIoU, 'b', label='Validation meanIoU')
    # plt.title('Training and validation meanIoU')
    # plt.legend()
    # plt.grid()
    # plt.savefig('bincrossentropy_meanIoU.jpg')
    # plt.show()