import numpy as np
import tensorflow as tf
import Preprocessing as pp
from keras.utils import to_categorical

np.random.seed(5)
tf.random.set_seed(5)

BATCH_SIZE = 10
NUM_EPOCHS = 30
IMG_SIZE = 180
Input_Shape = (IMG_SIZE, IMG_SIZE, 1)

patientImage_sp, patientLabel_sp = pp.load_Data(IMG_SIZE, 'Patient', 'Meander', 'Meander/')
healthyImage_sp, healthyLabel_sp = pp.load_Data(IMG_SIZE, 'Healthy', 'Meander', 'Meander/')
print("patientImage_Meander", len(patientImage_sp))
print("healthyImage_Meander", len(healthyImage_sp))

# Split Spiral
Xp_train_sp, Xp_test_sp, Yp_train_sp, Yp_test_sp = pp.split_data(patientImage_sp, patientLabel_sp, 0.15, 42)
Xh_train_sp, Xh_test_sp, Yh_train_sp, Yh_test_sp = pp.split_data(healthyImage_sp, healthyLabel_sp, 0.25, 42)
print("patientImage_train_Meander", len(Xp_train_sp))
print("healthyImage_train_Meander", len(Xh_train_sp))
print("healthyImage_test_Meander", len(Xh_test_sp))
print("patientImage_test_Meander", len(Xp_test_sp))

X_train =[]
X_test =[]
Y_train =[]
Y_test =[]

X_train.extend(Xp_train_sp)
X_train.extend(Xh_train_sp)
X_test.extend(Xp_test_sp)
X_test.extend(Xh_test_sp)
Y_train.extend(Yp_train_sp)
Y_train.extend(Yh_train_sp)
Y_test.extend(Yp_test_sp)
Y_test.extend(Yh_test_sp)

print("X_train",len(X_train))
print("X_test",len(X_test))

X_train, Y_train = pp.shuffleData(X_train,Y_train)

X_test = np.asarray(X_test)
y_test = np.asarray(Y_test)

Y_test = to_categorical(Y_test, num_classes=2)
train_model = pp.GoogleNet(Input_Shape)

train_model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

train_model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
train_score = train_model.evaluate(X_train,Y_train,verbose =0)
print("Train Score: ", train_score[0])
print("Train accuracy: ", train_score[1])
score = train_model.evaluate(X_test,Y_test,verbose =0)
print("Test Score: ", score[0])
print("Test accuracy: ", score[1])
# train_model.save('Mean_Alexnet.h5')
train_model.summary()

# Confusion Matrix
pp.ConfusionMatrix(train_model, X_test, Y_test)



