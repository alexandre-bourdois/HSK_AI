model.add(tf.keras.layers.Flatten(input_shape=(img_height, img_width, 3)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=178, activation=tf.nn.softmax))
