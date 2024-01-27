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

sample_size = 32
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
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(img_height, img_width, 3)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=178, activation=tf.nn.softmax))


# mnist= tf.keras.datasets.mnist
# (x_train, y_train), (x_text,y_test)= mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_text, axis=1)


# model=tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=1, validation_data=test_generator)

# accuracy, loss = model.evaluate(x_test, y_test)
# print(accuracy)
# print(loss)

model.save('digits.model')

# img= cv.imread('1.png')[:,:,0]
# img= np.invert(np.array([img]))
# prediction=model.predict(img)
# print(f'The result is probably:{ np.argmax(prediction)}')
# plt.imshow(img[0],cmap=plt.cm.binary)
# plt.show()