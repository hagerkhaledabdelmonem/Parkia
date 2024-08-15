import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
import Preprocessing as pp

np.random.seed(5)
tf.random.set_seed(5)

BATCH_SIZE = 32
NUM_EPOCHS = 60
IMG_SIZE = 180
Input_Shape = (IMG_SIZE, IMG_SIZE, 3)

# load spiral data
patientImage_sp, patientLabel_sp = pp.load_Data(IMG_SIZE, 'Patient', 'Spiral', 'Spiral/')
healthyImage_sp, healthyLabel_sp = pp.load_Data(IMG_SIZE, 'Healthy', 'Spiral', 'Spiral/')
print("patientImage_spiral", len(patientImage_sp))
print("healthyImage_spiral", len(healthyImage_sp))

# load Meander data
patientImage_m, patientLabel_m = pp.load_Data(IMG_SIZE, 'Patient', 'Meander', 'Meander/')
healthyImage_m, healthyLabel_m = pp.load_Data(IMG_SIZE, 'Healthy', 'Meander', 'Meander/')
print("patientImage_meander", len(patientImage_m))
print("healthyImage_meander", len(healthyImage_m))

# Split Spiral
Xp_train_sp, Xp_test_sp, Yp_train_sp, Yp_test_sp = pp.split_data(patientImage_sp, patientLabel_sp, 0.15, 42)
Xh_train_sp, Xh_test_sp, Yh_train_sp, Yh_test_sp = pp.split_data(healthyImage_sp, healthyLabel_sp, 0.25, 42)
print("patientImage_train_spiral", len(Xp_train_sp))
print("healthyImage_train_spiral", len(Xh_train_sp))
print("healthyImage_test_spiral", len(Xh_test_sp))
print("patientImage_test_spiral", len(Xp_test_sp))

# Split Meander
Xp_train_m, Xp_test_m, Yp_train_m, Yp_test_m = pp.split_data(patientImage_m, patientLabel_m, 0.15, 42)
Xh_train_m, Xh_test_m, Yh_train_m, Yh_test_m = pp.split_data(healthyImage_m, healthyLabel_m, 0.25, 42)
print("patientImage_train_meander", len(Xp_train_m))
print("healthyImage_train_meander", len(Xh_train_m))

# Circle
################ augmentation (Run the first time only )#####################################################
# image_files = glob.glob(os.path.join("Fusion_Circle_Augmented/HealthyCircle", '*.jpg'))
# for image in image_files:
#     os.remove(image)
#
# image_files = glob.glob(os.path.join("Fusion_Circle_Augmented/PatientCircle", '*.jpg'))
# for image in image_files:
#     os.remove(image)
# patientImage_c, patientLabel_c, patientName_c = pp.load_Circle_Data(IMG_SIZE, 'Patient', 'Circle', 'Circle/')
# healthyImage_c, healthyLabel_c, healthyName_c = pp.load_Circle_Data(IMG_SIZE, 'Healthy', 'Circle', 'Circle/')
# pp.augmentation(x_train=patientImage_c, y_train=patientLabel_c, save_dir="Fusion_Circle_Augmented/PatientCircle",imageName=patientName_c)
# pp.augmentation(x_train=healthyImage_c, y_train=healthyLabel_c, save_dir="Fusion_Circle_Augmented/HealthyCircle",imageName=healthyName_c)

# ######################################################################################

# Load Circle
patientImage_c, patientLabel_c = pp.load_Data(IMG_SIZE, 'Patient', 'Circle', 'Circle_AugmentationRGB_Last/')
healthyImage_c, healthyLabel_c = pp.load_Data(IMG_SIZE, 'Healthy', 'Circle', 'Circle_AugmentationRGB_Last/')

# Split Circle
Xp_train_c, Xp_test_c, Yp_train_c, Yp_test_c = pp.split_data(patientImage_c, patientLabel_c, 0.15, 42)
Xh_train_c, Xh_test_c, Yh_train_c, Yh_test_c = pp.split_data(healthyImage_c, healthyLabel_c, 0.25, 42)
print("CirclePatienttrain", len(Xp_train_c))
print("CircleHealthytrain", len(Xh_train_c))

# Healthy Train Data
Xh_train_sp = Xh_train_sp.reshape((Xh_train_sp.shape[0], IMG_SIZE, IMG_SIZE))
Xh_train_m = Xh_train_m.reshape((Xh_train_m.shape[0], IMG_SIZE, IMG_SIZE))
Xh_train_c = Xh_train_c.reshape((Xh_train_c.shape[0], IMG_SIZE, IMG_SIZE))
Xh_train = np.stack((Xh_train_sp, Xh_train_m, Xh_train_c), axis=-1)
Yh_train = Yh_train_m

# Patient Train Data
Xp_train_sp = Xp_train_sp.reshape((Xp_train_sp.shape[0], IMG_SIZE, IMG_SIZE))
Xp_train_m = Xp_train_m.reshape((Xp_train_m.shape[0], IMG_SIZE, IMG_SIZE))
Xp_train_c = Xp_train_c.reshape((Xp_train_c.shape[0], IMG_SIZE, IMG_SIZE))
Xp_train = np.stack((Xp_train_sp, Xp_train_m, Xp_train_c), axis=-1)
Yp_train = Yp_train_m

# concatenate healthy_train & patient_train
X_train = np.concatenate([Xh_train, Xp_train], axis=0)
Y_train = np.concatenate([Yh_train, Yp_train], axis=0)

# shuffle data
X_train_shuffle, Y_train_shuffle = pp.shuffleData(X_train , Y_train)

# Healthy Test Data
Xh_test_sp = Xh_test_sp.reshape((Xh_test_sp.shape[0], IMG_SIZE, IMG_SIZE))
Xh_test_m = Xh_test_m.reshape((Xh_test_m.shape[0], IMG_SIZE, IMG_SIZE))
Xh_test_c = Xh_test_c.reshape((Xh_test_c.shape[0], IMG_SIZE, IMG_SIZE))
Xh_test = np.stack((Xh_test_sp, Xh_test_m, Xh_test_c), axis=-1)
Yh_test = Yh_test_m

# Patient Test Data
Xp_test_sp = Xp_test_sp.reshape((Xp_test_sp.shape[0], IMG_SIZE, IMG_SIZE))
Xp_test_m = Xp_test_m.reshape((Xp_test_m.shape[0], IMG_SIZE, IMG_SIZE))
Xp_test_c = Xp_test_c.reshape((Xp_test_c.shape[0], IMG_SIZE, IMG_SIZE))
Xp_test = np.stack((Xp_test_sp, Xp_test_m, Xp_test_c), axis=-1)
Yp_test = Yp_test_m

# concatenate healthy_test & patient_test
X_test = np.concatenate([Xh_test, Xp_test], axis=0)
Y_test = np.concatenate([Yh_test, Yp_test], axis=0)
Y_test = to_categorical(Y_test)

# Model
train_model = pp.GoogleNet(Input_Shape)
train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.losses.BinaryCrossentropy(),
                    metrics=['accuracy'])
train_model.fit(X_train_shuffle, Y_train_shuffle, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
score = train_model.evaluate(X_test, Y_test, verbose=0)
print("Test Loss: ", score[0])
print("Test accuracy: ", score[1])

train_model.summary()
train_model.save('Save_Models/CNN_fusion4.h5')
pp.ConfusionMatrix(train_model, X_test, Y_test)


