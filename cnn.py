from __future__ import print_function
from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

data = np.load("C:/Users/zhong/PycharmProjects/kannadais/data/X_kannada_MNIST_train.npz")['arr_0']
x_new = np.load("C:/Users/zhong/PycharmProjects/kannadais/data/X_kannada_MNIST_test.npz")['arr_0']
Y = np.load("C:/Users/zhong/PycharmProjects/kannadais/data/y_kannada_MNIST_train.npz")['arr_0']
'''
sns.countplot(Y)
#  plt.show()
'''
x_train, x_test, y_train, y_test = train_test_split(data, Y, test_size=0.3, random_state=42, stratify=Y)
'''
import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(x_train[3],cmap='gray')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(x_train[8],cmap='gray')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(x_train[2],cmap='gray')
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(x_train[20],cmap='gray')
plt.show()
'''


# 数据预处理：能够喂入sequential网络
img_rows, img_cols = 28, 28
batch_size = 128
num_classes = 10
epochs = 5
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    x_new = x_new.reshape(x_new.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_new = x_new.reshape(x_new.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_new = x_new.astype('float32')
x_train /= 255
x_test /= 255
x_new /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#  构建sequential网络
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

# Compile model
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=3,
                                            verbose=1,
                                            factor=0.2,
                                            min_lr=0.00001)
# ImageDataGenerator用于生成假图像，扩充数据集
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)

# Model Training
hist = model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=20, validation_data=(x_test, y_test),
    verbose=2, steps_per_epoch=x_train.shape[0] // 128,
    callbacks=[learning_rate_reduction])

# Model Evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
