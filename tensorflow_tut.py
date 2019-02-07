import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# print("X test: ", x_test)
# print("y_test: ", y_test)
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def create_model():
  mnist = tf.keras.datasets.mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  normalized_x_train = tf.keras.utils.normalize(x_train, axis=1)
  normalized_x_test = tf.keras.utils.normalize(x_test, axis=1)
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
  model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
  model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(normalized_x_train, y_train, epochs=3)

  return model

x,y = x_test[0].shape
print("X: ", x, "Y: ", y)

# val_loss, val_acc = model.evaluate(x_test, y_test)

# model.save('tut_model.model')

# saver = tf.train.Saver()
# session = tf.Session()
# session.run(tf.global_variables_initializer())
# saver.save(session, 'tensor_data/foo_model')
# predictions = model.predict(x_test)


# print(val_loss, val_acc)