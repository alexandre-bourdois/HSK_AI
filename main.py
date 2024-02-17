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
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

def create_model(img_height, img_width, num_classes,train_generator,test_generator):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=num_classes, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator,validation_data=test_generator, epochs=3 )
    model.save('character.model')
    return model

def load_test_character(img_path):
    img = cv.imread(img_path)
    img = cv.resize(img, (67, 67))
    img = np.expand_dims(img, axis=0) 
    img = img / 255.0  
    return img

def predict_character():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((67, 67))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = list(train_generator.class_indices.keys())[predicted_class_index]
        prediction_label.config(text=f"Predicted character: {predicted_class_name}")
        img = Image.open(file_path)
        img= img.resize((500, 500))
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img

#Repertory of the dataset
train_data_dir = '../dataset_chinese_test/train'
test_data_dir = '../dataset_chinese_test/test'

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


# Create, compile and save the model
# model = create_model(img_height, img_width, num_classes,train_generator,test_generator)

# Load the trained model
model = tf.keras.models.load_model('character.model')

root = tk.Tk()
root.title("HSK Character Recognition App")

load_button = Button(root, text="Load Image", command=predict_character)
load_button.pack()

image_label = Label(root)
image_label.pack()

prediction_label = Label(root, text="")
prediction_label.pack()

root.mainloop()