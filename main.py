"""
HSK_AI
main.py

Created by Alexandre BOURDOIS.

"""
import tensorflow as tf
import scipy
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Function to create the model
def create_model(img_height, img_width, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=num_classes, activation=tf.nn.softmax)
    ])
    return model

#Repertory of the dataset
train_data_dir = '../dataset_chinese/train'
test_data_dir = '../dataset_chinese/test'

sample_size = 64
img_height, img_width = 67, 67 
num_classes=178

#Generate characters
train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=sample_size,
    class_mode='sparse'  
)

test_generator = test_data_gen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=sample_size,
    class_mode='sparse'
)

# Create and compile the model
model = create_model(img_height, img_width, num_classes)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator,validation_data=test_generator, epochs=1 )

model.save('numbers.model')

# Load the trained model
#model = tf.keras.models.load_model('numbers.model')


img_path = 'yi.png'
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (img_height, img_width))
img = np.invert(img) / 255.0  # Inverser l'image et normaliser les valeurs des pixels
img = np.expand_dims(img, axis=0)  # Ajouter une dimension pour le lot (batch)


prediction = model.predict(img)
print(f'The result is probably:{ np.argmax(prediction)}')
plt.imshow(img[0],cmap=plt.cm.binary)
plt.show()
