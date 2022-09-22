import tensorflow as tf
import numpy as np
import matplotlib, cv2
import matplotlib.pyplot as plt
import base64, io, os, time, gym
from gym.envs.tetris import tetris99
import functools
import time

# import tensorflow_probability as tfp
# import mitdeeplearning as mdl

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

env = tetris99.TetrisEnv()

print(env.observation_space.sample())