import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import ntpath
import cv2
import os
from itertools import chain
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, ZeroPadding2D
data_augment = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.2,
	brightness_range=[0.5, 1.5],
	rescale=1. / 255,
	fill_mode='nearest'
)

def load_Data(shape, IMG_SIZE):
	classes = ['Patient', 'Healthy']
	images = []
	labels = []
	for c in classes:
		try:
			for img in os.listdir(c + shape):
				img = cv2.imread(c + shape + '/' + img, 0)
				img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
				images.append(img)
				labels.append(["Healthy", "PD"][c == 'Patient'])
		except:
			pass
	data = pd.DataFrame({"images": images, "labels": labels})
	return data

def augmentation(x_train, y_train, save_dir):
	for index in range(len(x_train)):
		image_array = np.array(x_train[index])
		image_array = np.expand_dims(image_array, axis=-1)
		i = 0
		for batch in data_augment.flow(image_array, save_to_dir=save_dir, save_prefix=y_train[index],
									   save_format='jpg'):
			i += 1
			if i > 5:
				break

def load_data_file(dir, IMG_SIZE): #load file .npy
    loaded_data = np.load(f'{dir}.npy', allow_pickle=True)
    x, y = loaded_data[:, 0], loaded_data[:, 1]
    shape = x.shape
    x = np.array(list(chain.from_iterable(x)))
    x = np.array(list(chain.from_iterable(x)))
    x = x.reshape((shape[0], IMG_SIZE, IMG_SIZE, 1))
    return x, y

def save_data(data, directory): #save data in npy files
    imgset = np.array(data)
    dir = f"{directory}.npy"
    np.save(f"{directory}.npy", imgset)
    print("data saved successfully")

def load_all_data(save_dir):
    imagePaths = list(paths.list_images(save_dir))
    df = pd.DataFrame(columns=['images', 'labels'])
    for path in imagePaths:
        record = []
        record.append(cv2.imread(path, 0))
        head, image_name = ntpath.split(path)
        if image_name.startswith("Healthy"):
            record.append(1)
        else:
            record.append(0)
        df.loc[len(df.index)] = record

    df = df.sample(frac=1).reset_index(drop=True)
    return df

def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
  # Input:
  # - f1: number of filters of the 1x1 convolutional layer in the first path
  # - f2_conv1, f2_conv3 are number of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path
  # - f3_conv1, f3_conv5 are the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path
  # - f4: number of filters of the 1x1 convolutional layer in the fourth path

  # 1st path:
  path1 = Conv2D(filters=f1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)

  # 2nd path
  path2 = Conv2D(filters = f2_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path2 = Conv2D(filters = f2_conv3, kernel_size = (3,3), padding = 'same', activation = 'relu')(path2)

  # 3rd path
  path3 = Conv2D(filters = f3_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path3 = Conv2D(filters = f3_conv5, kernel_size = (5,5), padding = 'same', activation = 'relu')(path3)

  # 4th path
  path4 = MaxPooling2D((3,3), strides= (1,1), padding = 'same')(input_layer)
  path4 = Conv2D(filters = f4, kernel_size = (1,1), padding = 'same', activation = 'relu')(path4)

  output_layer = tf.keras.layers.concatenate([path1, path2, path3, path4], axis = -1)

  return output_layer

def GoogleNet(IMG_SIZE):
    input_layer = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # First convolutional layers
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Second convolutional layers
    x = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception modules
    x = Inception_block(x, 64, 96, 128, 16, 32, 32)
    x = Inception_block(x, 128, 128, 192, 32, 96, 64)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add more inception modules or other layers as needed
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(2, activation='softmax')(x)  # Assuming 2 classes

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def Alexnet_Model(IMG_SIZE):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (11, 11), strides=(4, 4), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Flatten(), # Flatten Layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),  # Adding dropout regularization
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Adding dropout regularization
        # Output Layer
        tf.keras.layers.Dense(2, activation='sigmoid')
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