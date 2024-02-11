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

def create_model(img_height, img_width, num_classes,train_generator,test_generator):
    train_generator=tf.keras.utils.normalize(train_generator, axis=1)
    test_generator=tf.keras.utils.normalize(test_generator, axis=1)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(img_height, img_width,      1)),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=num_classes, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator,validation_data=test_generator, epochs=2 )
    model.save('character.model')
    return model

def load_test_character(img_path):
    img = cv.imread(img_path)
    img = cv.resize(img, (67, 67))
    img = np.expand_dims(img, axis=0) 
    img = img / 255.0  
    return img

#Repertory of the dataset
train_data_dir = '../dataset_chinese/train'
test_data_dir = '../dataset_chinese/test'

sample_size = 32
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


# Create, compile and save the model
# model = create_model(img_height, img_width, num_classes,train_generator,test_generator)

# Load the trained model
model = tf.keras.models.load_model('character.model')


img = load_test_character('yi.png')


prediction = model.predict(img)
predicted_class_index = np.argmax(prediction)
predicted_class_name = list(train_generator.class_indices.keys())[predicted_class_index]
print(f"The predicted character is: {predicted_class_name}")
plt.imshow(img[0],cmap=plt.cm.binary)
plt.show()
