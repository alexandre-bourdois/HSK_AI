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
from PIL import ImageGrab
import cv2
from tkinter import filedialog, Label, Button,Canvas
from PIL import Image, ImageTk

def create_model(img_height, img_width, num_classes, train_generator, test_generator):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5), 
        tf.keras.layers.Dense(units=num_classes, activation=tf.nn.softmax)
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, validation_data=test_generator, epochs=10)
    model.save('character.model')
    return model

class DrawingApp:
    def __init__(self, master, predict_function):
        self.master = master
        self.predict_function = predict_function
        self.setup_ui()
        self.setup_bindings()

    def setup_ui(self):
        self.canvas = Canvas(self.master, width=200, height=200, bg="white")
        self.canvas.pack()

        self.clear_button = Button(self.master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.predict_button = Button(self.master, text="Predict", command=self.predict_character)
        self.predict_button.pack()

        self.save_button = Button(self.master, text="Save", command=self.save_image)
        self.save_button.pack()
    
    def save_image(self):
        image = self.get_image()
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        if file_path:
            cv2.imwrite(file_path, image)

    def setup_bindings(self):
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def clear_canvas(self):
        self.canvas.delete("all")

    def draw(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)

    def reset(self, event):
        self.canvas.update()

    def predict_character(self):
        # Convert the drawn image to grayscale
        drawn_image = self.get_image()

        # Resize the image to match the expected input shape of the model
        resized_image = cv2.resize(drawn_image, (67, 67))

        # Check if the image is in grayscale or RGB
        if len(resized_image.shape) == 2:  # Grayscale image
            resized_image = np.expand_dims(resized_image, axis=2)  # Add channel dimension
            resized_image = np.repeat(resized_image, 3, axis=2)  # Convert to RGB
        elif resized_image.shape[2] == 4:  # RGBA image
            resized_image = resized_image[:, :, :3]  # Remove alpha channel

        img = np.expand_dims(resized_image, axis=0)
        img = img / 255.0

        # Make prediction
        prediction = self.predict_function(img)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = list(train_generator.class_indices.keys())[predicted_class_index]
        self.display_prediction(predicted_class_name)

    def get_image(self):
        self.master.update()
        x0 = self.master.winfo_rootx() + self.canvas.winfo_x()
        y0 = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x0 + self.canvas.winfo_width()
        y1 = y0 + self.canvas.winfo_height()
        image = ImageGrab.grab().crop((x0, y0, x1, y1))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def display_prediction(self, predicted_class_name):
        self.clear_canvas()
        self.canvas.create_text(100, 100, text=f"Predicted character: {predicted_class_name}", font=("Helvetica", 12))

def predict_character(img):
    prediction = model.predict(img)
    return prediction

#Repertory of the dataset
train_data_dir = '../dataset_chinese_test/train'
test_data_dir = '../dataset_chinese_test/test'

sample_size = 600
img_height, img_width = 67, 67 
num_classes=5

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
#model = create_model(img_height, img_width, num_classes,train_generator,test_generator)

# Load the trained model
model = tf.keras.models.load_model('character.model')


root = tk.Tk()
root.title("HSK Character Recognition App")

drawing_app = DrawingApp(root, predict_character)

root.mainloop()
