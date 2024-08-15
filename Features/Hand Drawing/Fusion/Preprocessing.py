from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,Input, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D,Reshape
import pandas as pd
import numpy as np
import cv2
import os
from itertools import chain
from keras_preprocessing.image import ImageDataGenerator
from keras import layers

def load_Data(IMG_SIZE, classes: str, shape_name, path):
    images = []
    labels = []
    cl = classes + shape_name
    for img in os.listdir(os.path.join(path, cl)):
        img_path = os.path.join(path, cl, img)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append([0, 1][classes == 'Patient'])

    images = np.array(images, dtype='float32') / 255.
    labels = np.array(labels)
    return images, labels

def load_Circle_Data(IMG_SIZE, classes: str, shape_name, path):
    images = []
    labels = []
    cl = classes + shape_name
    Image_name = []
    for img in os.listdir(os.path.join(path, cl)):
        parts = img.split('-')
        desired_part = parts[1].split('.')[0]
        Image_name.append(desired_part)
        img_path = os.path.join(path, cl, img)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append([0, 1][classes == 'Patient'])

    images = np.array(images, dtype='float32')
    labels = np.array(labels)
    return images, labels, Image_name

def shuffleData(X_train_data, Y_train_data):
    X_train = []
    Y_train = []

    X_train.extend(X_train_data)
    Y_train.extend(Y_train_data)
    data = pd.DataFrame({"images": X_train, "labels": Y_train})
    df_shuffled = data.sample(frac=1).reset_index(drop=True)
    x = np.array(df_shuffled.drop('labels', axis=1))
    new_X_train = np.array(list(chain.from_iterable(x)))
    new_Y_train = np.array(df_shuffled['labels']).reshape(-1, 1).ravel()
    new_Y_train = to_categorical(new_Y_train, num_classes=2)
    return new_X_train, new_Y_train

def data(Xp_train, Xh_train, Xp_test, Xh_test, Yp_train, Yh_train, Yp_test, Yh_test):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    X_train.extend(Xp_train)
    X_train.extend(Xh_train)
    X_test.extend(Xp_test)
    X_test.extend(Xh_test)
    Y_train.extend(Yp_train)
    Y_train.extend(Yh_train)
    Y_test.extend(Yp_test)
    Y_test.extend(Yh_test)

    data = pd.DataFrame({"images": X_train, "labels": Y_train})
    df_shuffled = data.sample(frac=1).reset_index(drop=True)
    x = np.array(df_shuffled.drop('labels', axis=1))
    X_train = np.array(list(chain.from_iterable(x)))
    Y_train = np.array(df_shuffled['labels']).reshape(-1, 1).ravel()
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)
    Y_train = to_categorical(Y_train, num_classes=2)
    Y_test = to_categorical(Y_test, num_classes=2)

    return X_train, Y_train, X_test, Y_test

def split_data(data, labels, testSize, random_state):
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=testSize, random_state=random_state,
                                                        shuffle=False)
    return X_train, X_test, Y_train, Y_test

def CNN_Model(Input_Shape):

    regularizers = keras.regularizers.l2(1e-3)
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(32, (3, 3), input_shape=Input_Shape, activation="relu", kernel_regularizer=regularizers))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=regularizers))
    model.add(keras.layers.MaxPooling2D(2, 2))
    model.add(keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=regularizers))
    model.add(keras.layers.MaxPooling2D(2, 2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2, activation="softmax"))

    return model

def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
        # Input:
        # - f1: number of filters of the 1x1 convolutional layer in the first path
        # - f2_conv1, f2_conv3 are number of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path
        # - f3_conv1, f3_conv5 are the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path
        # - f4: number of filters of the 1x1 convolutional layer in the fourth path

    # 1st path:
    path1 = Conv2D(filters=f1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # 2nd path
    path2 = Conv2D(filters=f2_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=f2_conv3, kernel_size=(3, 3), padding='same', activation='relu')(path2)

    # 3rd path
    path3 = Conv2D(filters=f3_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=f3_conv5, kernel_size=(5, 5), padding='same', activation='relu')(path3)

    # 4th path
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(filters=f4, kernel_size=(1, 1), padding='same', activation='relu')(path4)

    output_layer = tf.keras.layers.concatenate([path1, path2, path3, path4], axis=-1)

    return output_layer

def GoogleNet(Input_Shape):
    input_layer = tf.keras.layers.Input(shape=Input_Shape)

    # First convolutional layers
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Second convolutional layers
    x = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception modules
    x = Inception_block(x, 64, 96, 128, 16, 32, 32)
    x = Inception_block(x, 128, 128, 192, 32, 96, 64)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(2, activation='softmax')(x)  # Assuming 2 classes

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def Alexnet_Model(Input_Shape):

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=Input_Shape, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(activation='relu', units=128),
        layers.Dense(2, activation='sigmoid'),
    ])

    return model


def ConfusionMatrix(model, X_test, Y_test):
    test_integer_labels = np.argmax(Y_test, axis=1) # If your test_labels are one-hot encoded, convert them to integer labels
    predictions = model.predict(X_test)  # Get model predictions
    predicted_labels = np.argmax(predictions, axis=1)
    cm = confusion_matrix(test_integer_labels, predicted_labels) # Compute the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

##### augmentatin#######
data_augment = ImageDataGenerator(
		rotation_range=20,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest'
)

def augmentation(x_train, y_train, save_dir, imageName):
	counter = 0
	ch = ""
	counter_to_char = {0: "B", 1: "C", 2: "D"}

	while counter < 3:
		if counter in counter_to_char:  # Check if the counter value is in the dictionary
			ch = counter_to_char[counter]
		for index in range(len(x_train)):
			image_array = np.array(x_train[index])
			image_array = np.expand_dims(image_array, axis=-1)
			image_array = image_array.reshape(1, image_array.shape[0], image_array.shape[1], image_array.shape[2])
			# n = y_train[index]
			save_name = f"circ{ch}-{imageName[index]}" if y_train[index] == 1 else f"circ{ch}-{imageName[index]}"
			for batch in data_augment.flow(image_array, save_to_dir=save_dir, save_prefix=save_name,
										   save_format='jpg'): break
		counter += 1






