import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import os
import time
import matplotlib.pyplot as plt
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications import MobileNet
import pandas as pd
import keras_metrics as km

start = time.time()
PATH = os.path.join('casme_ROI')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'test')

train_anger_dir = os.path.join(train_dir, 'disgust')
train_contempt_dir = os.path.join(train_dir, 'fear')
train_disgust_dir = os.path.join(train_dir, 'happiness')
train_fear_dir = os.path.join(train_dir, 'others')
train_happy_dir = os.path.join(train_dir, 'repression')
train_sadness_dir = os.path.join(train_dir, 'sadness')
train_surprise_dir = os.path.join(train_dir, 'surprise')

validation_anger_dir = os.path.join(validation_dir, 'disgust')
validation_contempt_dir = os.path.join(validation_dir, 'fear')
validation_disgust_dir = os.path.join(validation_dir, 'happiness')
validation_fear_dir = os.path.join(validation_dir, 'others')
validation_happy_dir = os.path.join(validation_dir, 'repression')
validation_sadness_dir = os.path.join(validation_dir, 'sadness')
validation_surprise_dir = os.path.join(validation_dir, 'surprise')

num_anger_tr = len(os.listdir(train_anger_dir))
num_contempt_tr = len(os.listdir(train_contempt_dir))
num_disgust_tr = len(os.listdir(train_disgust_dir))
num_fear_tr = len(os.listdir(train_fear_dir))
num_happy_tr = len(os.listdir(train_happy_dir))
num_sadness_tr = len(os.listdir(train_sadness_dir))
num_surprise_tr = len(os.listdir(train_surprise_dir))

num_anger_val = len(os.listdir(validation_anger_dir))
num_contempt_val = len(os.listdir(validation_contempt_dir))
num_disgust_val = len(os.listdir(validation_disgust_dir))
num_fear_val = len(os.listdir(validation_fear_dir))
num_happy_val = len(os.listdir(validation_happy_dir))
num_sadness_val = len(os.listdir(validation_sadness_dir))
num_surprise_val = len(os.listdir(validation_surprise_dir))

total_train = num_anger_tr + num_contempt_tr + num_disgust_tr + \
    num_fear_tr + num_happy_tr + num_sadness_tr + num_surprise_tr
total_val = num_anger_val + num_contempt_val + num_disgust_val + \
    num_fear_val + num_happy_val + num_sadness_val + num_surprise_val

batch_size = 32
epochs = 100
IMG_HEIGHT = 48
IMG_WIDTH = 48

train_image_generator = ImageDataGenerator(rescale=1. / 255)

validation_image_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(
    directory=train_dir, shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH))
val_data_gen = validation_image_generator.flow_from_directory(
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH))


def big_Xception(input_shape, num_classes, l2_regularization=0.01):

    # base
    img_input = Input(input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model = Model(img_input, output)
    return model
#
# def myMobileNet(input_shape,num_classes):
#     """
#     我们采用函数的方法得到需要的模型
#     这里我们采用的是在ImageNet上进行预训练过的权重，这个权重经过1000分类，包含了常见的物体识别，因此可以得到比较好的效果，
#     同时可以大大缩短我们自己训练时的耗时，在短时间内将准确率提升到一个比较高的水平
#     :return: 返回得到一个利用函数方法构建出来的模型，其中卷积层采用的是vgg自带的权重
#     """
#     # 设置输入层，作为图像数据输入
#     inputs = tf.keras.layers.Input(shape=input_shape)
#     # 导入预训练模型，include_top=False代表自己重新写输出层
#     myMobileNet = MobileNet(input_shape=input_shape, include_top=False)
#     # 将预训练模型的每一层都设置为不可训练，此处我们暂时只训练全连接层，卷积层暂时不管，后期等准确率到达比较高的水平时，再设置为可训练
#     # 我们可以更改设置，为True
#
#     for layer in myMobileNet.layers:
#         myMobileNet.trainable = False
#
#
#     # 连接上我们自己的全连接层，作为模型的训练目标
#     # 由于全连接层属于参数非常密集的层，因此需要进行一定程度的正则化，对输出每一层的连接进行限制，减缓过拟合现象的发声
#     x = big_Xception(inputs)
#     x = layers.Flatten()(x)
#     x = layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.512))(x)
#     x = layers.Dropout(rate=0.4)(x)
#     x = layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.128))(x)
#     x = layers.Dropout(rate=0.2)(x)
#     outputs = layers.Dense(num_classes, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
#
#     # 返回一个keras的模型
#     return tf.keras.Model(inputs=inputs, outputs=outputs)


input_shape = (48, 48, 3)
num_classes = 7
patience = 50
base_path = 'models/'

model = big_Xception(input_shape, num_classes)
model.compile(optimizer='adam',  # 优化器采用adam
              loss='categorical_crossentropy',  # 多分类的对数损失函数
              metrics=['accuracy', km.f1_score(), km.precision(), km.recall()])
model.summary()

log_file_path = base_path + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience / 4),
                              verbose=1)
# 模型位置及命名
trained_models_path = base_path + 'big_Xception(CASME_ROI)'
model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.hdf5'

# 定义模型权重位置、命名等
model_checkpoint = ModelCheckpoint(model_names,
                                   'val_loss', verbose=1,
                                   save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
    callbacks=callbacks,
)


data = pd.DataFrame(history.history)
data.insert(0, "epoch", history.epoch)
# data.insert(1,"spendTime",time_callback.times)
data.to_excel('big_Xception.xlsx', float_format="%.4f", index=False)

# end = time.time()
model.summary()
#
# end = time.time()
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(
#     len(acc)
# )
# print("This  %d epochs cost time: %f s , average %f s per epoch" % (len(acc), (end - start), (end - start) / epochs))
#
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.savefig('accuracy(ROI).png')
#
# # 记录模型优化过程及准确率
# logFilePath = 'checkpoint/accuracy_and_loss(ROI).txt'
# if os.path.isfile(logFilePath):
#     print("Log file exists.")
#     logWriter = open(logFilePath, 'a')
# else:
#     print("Log file does not exists. Make it .")
#     logWriter = open(logFilePath, 'w')
#
# logWriter.write('Training finish at : ' + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '\n')
# logWriter.write('model id : ' + str(model.name) + '\n')
# logWriter.write('Total epoch : ' + str(epochs) + '\n')
# logWriter.write('These epochs cost time (second) : ' + str((end - start)) + '\n')
# logWriter.write('Training accuracy  : ' + '\n          ' + str(history.history['accuracy']) + '\n')
# logWriter.write('Validation accuracy: ' + '\n          ' + str(history.history['val_accuracy']) + '\n')
# logWriter.write('---------------------------------------------------------------------------\n')
# logWriter.write('\n')
# print("Log file has successfully written down.")
# logWriter.close()
