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

#Repertory of the dataset
train_data_dir = '../dataset_chinese/train'
test_data_dir = '../dataset_chinese/test'

sample_size = 64
img_height, img_width = 67, 67 

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

# create model
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(img_height, img_width, 3)))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=178, activation=tf.nn.softmax))

# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(train_generator,validation_data=test_generator, epochs=1 )

# accuracy, loss = model.evaluate(train_generator, test_generator)
# print(accuracy)
# print(loss)

# model.save('digits.model')

# Charger une image et la pretraiter
img_path = 'bu.png'
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (img_height, img_width))
img = np.invert(img) / 255.0  # Inverser l'image et normaliser les valeurs des pixels
img = np.expand_dims(img, axis=0)  # Ajouter une dimension pour le lot (batch)


model = tf.keras.models.load_model('digits.model')

prediction = model.predict(img)
print(f'The result is probably:{ np.argmax(prediction)}')
plt.imshow(img[0],cmap=plt.cm.binary)
plt.show()
