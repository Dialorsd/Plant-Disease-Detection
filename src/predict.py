import json
import numpy as np
import tensorflow as tf
from src import config as C


def load():
    model = tf.keras.models.load_model(C.MODEL_PATH)
    classes = json.loads(C.CLASSES_PATH.read_text())
    return model, classes


def predict(img, model, classes):
    img = img.convert("RGB").resize(C.IMG_SIZE)
    x = np.expand_dims(np.array(img), 0)
    probs = model.predict(x, verbose=0)[0]
    i = int(probs.argmax())
    return classes[i], float(probs[i])
