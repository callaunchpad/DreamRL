import numpy as np
import sys

import keras
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Lambda, Input, Dense

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from os import sys
import argparse
import os

sys.path.insert(0, 'data')
from extract_img_action import extract
sys.path.insert(0, 'vae-cnn')
from encode_images_func import encode
from vae import VAE
sys.path.insert(0, 'mdn-rnn')
from mdn import MDNRNN

print("Extracting path...")
img_path_name, action_path_name = extract("LunarLander-v2", 2500, 150, False, 80)

# training VAE

print("Training VAE...")
convVae = VAE()
convVae.make_vae(img_path_name + ".npz", 2)
convVae.model_name = 'models/LunarLander_vae_64.h5'
convVae.epochs = 1000
convVae.train_vae()

encode(img_path_name, 'vae-cnn/LunarLander_64.h5', 64, False)
latent_path_name = img_path_name + '_latent.npz'

latent = np.load(latent_path_name) # (2500, ?, 64)
act = np.load(action_path_name + '.npz') # (2500, ?, 1)

combined_input = []
combined_output = []

def hot(tot, i):
    v = np.zeros(tot)
    v[i] = 1
    return v

print("Saving output...")
for f in latent.files:
    c = np.concatenate([latent[f], np.array([hot(4, i) for i in act[f]])], axis=1)
    missing = 151 - c.shape[0]
    c = np.concatenate([c, np.zeros((missing, 68))], axis=0)
    combined_input.append(c[:-1])
    combined_output.append(c[1:, :-4])

np.save('LunarLander_MDN_in', combined_input)
np.save('LunarLander_MDN_out', combined_output)

# training MDN

# MDN Parameters
print("Configuring MDN...")
hps = {}
hps['batch_size'] = 5
hps['max_seq_len'] = 150
hps['in_width'] = 68 # latent + action
hps['out_width'] = 64 # Latent
hps['action_size'] = 4 # in width - out width
hps['rnn_size'] = 128
hps['kmix'] = 5
hps['dropout'] = 0.5
hps['recurrent_dropout'] = 0.5
hps['validation_split'] = 0.1
hps['epochs'] = 24

mdnrnn = MDNRNN(hps)
print("Finished building MDN, starting training...")
X = np.load('LunarLander_MDN_in.npy')
Y = np.load('LunarLander_MDN_out.npy')

mdnrnn.train(X, Y)
print("Finished training mdn")
mdnrnn.save('models/LunarLander_vae_64')
