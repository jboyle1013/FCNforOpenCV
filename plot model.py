import tensorflow as tf
from model import FCN_model
from generator import Generator
import os
import matplotlib.pyplot as plt

history = tf.keras.models.load_model('DiceFCN.keras')


