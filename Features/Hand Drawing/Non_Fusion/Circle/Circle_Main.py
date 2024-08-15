import tensorflow as tf
from keras.callbacks import Callback
from keras.utils import to_categorical
import Circle_Preprocessing as pp
import os
import glob
from sklearn.model_selection import train_test_split

IMG_SIZE = 180

class CustomEarlyStopping(Callback): #custom early stopping
    def __init__(self, min_loss):
        super(CustomEarlyStopping, self).__init__()
        self.min_loss = min_loss

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss is not None and current_loss < self.min_loss:
            print(f"\nReached minimum loss ({self.min_loss}). Stopping training.")
            self.model.stop_training = True

############################## augmentation (Run the first time only )######################################
data_circle = pp.load_Data("Circle", IMG_SIZE)
X = data_circle.drop('labels',axis=1)
Y = data_circle['labels']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=45)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, shuffle=True, random_state=43)

image_files = glob.glob(os.path.join("Train_Non_Fusion_Circle_Augmented", '*.jpg'))
for image in image_files:
    os.remove(image)

pp.augmentation(x_train=X_train.values.tolist(), y_train=Y_train.tolist(), save_dir="Train_Non_Fusion_Circle_Augmented")
train_data = pp.load_all_data("Train_Non_Fusion_Circle_Augmented")

X_test['labels'] = Y_test
X_val['labels'] = Y_val

pp.save_data(train_data,"train")
pp.save_data(X_test,"test")
pp.save_data(X_val,"validation")
##################################################################################
#load data
X_train, Y_train = pp.load_data_file("train",IMG_SIZE)
X_test, Y_test = pp.load_data_file("test", IMG_SIZE)
X_val, Y_val = pp.load_data_file("validation", IMG_SIZE)
Y_test[Y_test == 'Healthy'] = 1
Y_test[Y_test == 'PD'] = 0
Y_val[Y_val == 'Healthy'] = 1
Y_val[Y_val == 'PD'] = 0
Y_train = to_categorical(Y_train, num_classes=2)
Y_test = to_categorical(Y_test, num_classes=2)
Y_val = to_categorical(Y_val, num_classes=2)

# Model
alexNet_model = pp.Alexnet_Model(IMG_SIZE)
custom_early_stopping = CustomEarlyStopping(min_loss=0.03)
alexNet_model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
history = alexNet_model.fit(X_train,Y_train,epochs=35
                            ,batch_size=8,verbose=1)
score_train = alexNet_model.evaluate(X_train, Y_train)
score_test = alexNet_model.evaluate(X_test, Y_test)
score_val = alexNet_model.evaluate(X_val, Y_val)

print("Train Score: ", score_train[0])
print("Train accuracy: ", score_train[1] * 100)
print("validation Score: ", score_val[0])
print("validation accuracy: ", score_val[1] * 100)
print("Test Score: ", score_test[0])
print("Test accuracy: ", score_test[1] * 100)

#Confusion Matrix
pp.ConfusionMatrix(alexNet_model, X_test, Y_test)

#alexNet_model.save("AlexNetModel.h5") # save AlexNet Model


