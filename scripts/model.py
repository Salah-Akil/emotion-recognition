import tensorflow as tf
import numpy as np
from keras.models import load_model
import os


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Model(object):

    EMOTIONS_LIST_CK = ["Angry", "Contempt", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
    EMOTIONS_LIST_KAGGLE = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model_file):
        self.model = load_model(model_file)

    def predict_emotion_kaggle(self,img):
        self.predictions = self.model.predict(img)
        return Model.EMOTIONS_LIST_KAGGLE[np.argmax(self.predictions)]

    def predict_emotion_ck(self,img):
        self.predictions = self.model.predict(img)
        return Model.EMOTIONS_LIST_CK[np.argmax(self.predictions)]