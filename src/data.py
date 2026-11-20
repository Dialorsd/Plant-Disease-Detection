import tensorflow as tf
from src import config as C


def load_datasets():
    train = tf.keras.utils.image_dataset_from_directory(
        C.DATA_DIR, validation_split=0.2, subset="training", seed=42,
        image_size=C.IMG_SIZE, batch_size=C.BATCH_SIZE)
    val = tf.keras.utils.image_dataset_from_directory(
        C.DATA_DIR, validation_split=0.2, subset="validation", seed=42,
        image_size=C.IMG_SIZE, batch_size=C.BATCH_SIZE)
    return train, val, train.class_names
