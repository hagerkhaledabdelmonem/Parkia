from keras.models import load_model
import numpy as np
import cv2

Model = load_model('Save_Models/CNN_fusion98.1oldCNN_arc.h5')

# Path to the image file
sp_Image_Path ='Spi_L_N.jpg'
M_Image_Path ='Mean_L_N.jpg'
C_Image_Path ='Cir_L_N.jpg'
arr_path = [sp_Image_Path, M_Image_Path, C_Image_Path]
IMG_SIZE =180

# Read and preprocess the image
image = []
for path in arr_path:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    image.append(img)

image_fusion = np.stack((image[0], image[1], image[2]), axis=-1)
image_fusion = np.expand_dims(image_fusion, axis=0)
# Make predictions
predictions = Model.predict(image_fusion)
print(predictions)