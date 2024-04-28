# import os os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Isaac Ulises Ascencio Padilla

# Librerias
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

# Preprocesado
TrainImages = ImageDataGenerator(
    rescale= 1./255,
    shear_range= 0.2,
    zoom_range= 0.2,
    rotation_range= 20,
    horizontal_flip= True
)

TestImages = ImageDataGenerator(
    rescale= 1./255,
)

TrainSet = TrainImages.flow_from_directory(
    'datasets/chest_xray', #Ruta hacia el dataset
    target_size= (256,256),
    batch_size= 25,
    class_mode= 'binary'
)

TestSet = TestImages.flow_from_directory(
    'datasets/chest_xray', #Ruta hacia el dataset
    target_size= (256,256),
    batch_size= 25,
    class_mode= 'binary'
)

cnn = Sequential()

# Convolución
cnn.add (
    Conv2D (
        filters= 32,
        kernel_size= (5,5),
        activation= "relu", #Subcapa relu
        input_shape = (256,256,3)
    )
)

# Pooling
cnn.add (
    MaxPool2D(pool_size= (2,2))
)

# Flaten
cnn.add (
    Flatten()
)

# Full connection
cnn.add(
    Dense(units= 128, 
          activation= "relu")
)

# Salida
cnn.add(
    Dense(
        units= 1,
        activation= "sigmoid"
    )
)

# Compile
cnn.compile(optimizer= "adam",
            loss= "binary_crossentropy",
            metrics= ["accuracy"]
            )

# Entrenamiento
cnn.fit(
    TrainSet,
    epochs= 5,
    steps_per_epoch= len(TrainSet),
    validation_data= TrainSet,
    validation_steps= len(TestSet)
)


from keras.models import load_model
cnn.save('cnn.h5')

# Visualizar la arquitectura de la red neuronal
from keras.utils import plot_model
plot_model(cnn,to_file='cnn.png',show_shapes=True,show_layer_activations=True,show_layer_names=True) 


import cv2
import numpy as np
import os



def preprocess_image(image_path, target_size=(256, 256)):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Error reading image: {image_path}")

    image = cv2.resize(image, target_size)

    image = image.astype('float32') / 255.0

    image = np.expand_dims(image, axis=0)

    return image

model = load_model('cnn.h5')

image_folder = "datasets\chest_xray\predict"


for image_filename in os.listdir(image_folder):
    if image_filename.endswith('.jpg') or image_filename.endswith('.jpeg'):
        image_path = os.path.join(image_folder, image_filename)

        try:
            image_array = preprocess_image(image_path)

            prediction = model.predict(image_array)
            prediction_label = "Pneumonia" if prediction[0][0] < 0.90 else "Normal"  
            # prediction_label = "Pneumonia" if prediction[0][0] >= 0.5 else "Normal"

            print(f"Image: {image_filename} - Prediction: {prediction_label}")
            print(prediction)

        except ValueError as e:
            print(f"Error processing image: {image_filename} - {e}")



try:
    image_array = preprocess_image(image_path)
    prediction = model.predict(image_array)

except ValueError as e:
    print(f"Error: {e}")

