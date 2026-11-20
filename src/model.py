import tensorflow as tf
from src import config as C


def build_model(num_classes):
    L = tf.keras.layers
    model = tf.keras.Sequential([
        L.Input(shape=C.IMG_SIZE + (3,)),
        L.Rescaling(1 / 255.),
        L.Conv2D(32, 3, activation="relu"),
        L.MaxPooling2D(),
        L.Conv2D(64, 3, activation="relu"),
        L.MaxPooling2D(),
        L.Flatten(),
        L.Dense(128, activation="relu"),
        L.Dense(num_classes, activation="softmax"),
    ])
    model.compile("adam", "sparse_categorical_crossentropy", ["accuracy"])
    return model
