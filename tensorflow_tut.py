import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalized_x_train = tf.keras.utils.normalize(x_train, axis=1)
# normalized_x_test = tf.keras.utils.normalize(x_test, axis=1)
# print("X test: ", x_test)
# print("y_test: ", y_test)

def create_model():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
  model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
  model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

# model = create_model()
# model.fit(normalized_x_train, y_train, epochs=3)

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    tf.gfile.GFile()
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

graph = load_graph
val_loss, val_acc = model.evaluate(x_test, y_test)

# model.save('tut_model.model')

saver = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())
saver.save(session, 'tensor_data/foo_model')
predictions = model.predict(x_test)


print(val_loss, val_acc)