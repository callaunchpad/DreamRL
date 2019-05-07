import os
import numpy as np
import sys
import json
import argparse
import gym
import imageio

sys.path.insert(0, 'data')
from extract_img_action import extract, get_path_names, compress_image
from action_utils import ActionUtils, one_hot
sys.path.insert(0, 'vae-cnn')
from vae_process_images import vae_process_images
from vae import VAE
sys.path.insert(0, 'mdn-rnn')
from mdn import MDNRNN

# Set up Paths
# Lunar Lander
# MDNRNN_PATH = "model_weights/LunarLander_MDN_test.h5"
# VAE_PATH = "model_weights/LunarLander_VAE_test.h5"
# CONFIG_PATH = "configs/LunarLander-test.json"

# Space Invaders
MDNRNN_PATH = "model_weights/SpaceInvaders_MDN_test.h5"
VAE_PATH = "model_weights/SpaceInvaders_64.h5"
CONFIG_PATH = "configs/SpaceInvaders-test.json"

def dream_vis(env, mdnrnn, vae, params, mdn_hps, name, seq_length=25, skip_actions=30):
    observation = env.reset()
    # Run a few steps of the game
    state = mdnrnn.rnn_init_state()
    action = env.action_space.sample()
    for _ in range(skip_actions):
        img = env.render(mode='rgb_array')
        img = compress_image(img, size=params['img_size'])
        z = vae.encode_image(np.array([img]))[0]
        state = mdnrnn.rnn_next_state(z, one_hot(params['action_size'], action), state)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

    # Get the initial Z Vector
    img = env.render(mode='rgb_array')
    img = compress_image(img, size=params['img_size'])
    init_z = vae.encode_image(np.array([img]))[0]

    # Run In Dream
    actions = []
    for _ in range(seq_length):
        action = env.action_space.sample()
        actions.append(one_hot(params['action_size'], action))
        observation, reward, done, info = env.step(action)
    actions = np.array(actions)

    zs = mdnrnn.sample_sequence(init_z, actions, length=seq_length, prev_state=state)
    print(zs.shape, params['latent_size'])
    reconstructed_imgs = np.array([vae.decode_latent(np.array([l]))[0] for l in zs])

    # Get DAT GIF
    seq = (reconstructed_imgs * 255).astype(int)
    imageio.mimsave(name + '.gif', seq)
    print("Done with dream gif", name)

def frame_vis(env, mdnrnn, vae, params, mdn_hps, name, skip_actions=25):
    observation = env.reset()
    # Run a few steps of the game
    state = mdnrnn.rnn_init_state()
    action = env.action_space.sample()
    for _ in range(skip_actions):
        img = env.render(mode='rgb_array')
        img = compress_image(img, size=params['img_size'])
        z = vae.encode_image(np.array([img]))[0]
        state = mdnrnn.rnn_next_state(z, one_hot(params['action_size'], action), state)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
    
    original_img = env.render(mode='rgb_array')
    original_img = compress_image(original_img, size=params['img_size'])
    init_z = vae.encode_image(np.array([original_img]))[0]

    action = env.action_space.sample()
    one_hot_action = one_hot(params['action_size'], action)

    z, state = mdnrnn.sample_z(init_z, one_hot_action, state)

    env.step(action)
    new_img = env.render(mode='rgb_array')
    new_img = compress_image(new_img, size=params['img_size'])

    new_img_reconstruct = vae.decode_latent(z[0])[0]

    imageio.imsave(name + '_orig.jpg', (new_img * 255).astype(int))
    imageio.imsave(name + '_mdn.jpg', (new_img_reconstruct * 255).astype(int))    
    
def main():
    # Load the HyperParameters
    params = json.load(open(CONFIG_PATH))[0]
    utils = ActionUtils(params['env_name'])
    action_size = utils.action_size()
    params['action_size'] = action_size
    mdn_hps = params['mdn_hps']
    mdn_hps['max_seq_len'] = params['max_seq_len']
    mdn_hps['in_width'] = params['latent_size'] + action_size
    mdn_hps['out_width'] = params['latent_size']
    mdn_hps['action_size'] = action_size
    mdn_hps['rnn_size'] = params['hidden_size']
    mdn_hps = MDNRNN.set_hps_to_inference(mdn_hps)

    # Create the MDN and load the params
    mdnrnn = MDNRNN(mdn_hps)
    mdnrnn.load(MDNRNN_PATH)

    # Create the VAE
    vae = VAE()
    vae.make_vae_shape(params['img_size'], params['img_size'], params['latent_size'])
    vae.load_model(VAE_PATH)

    # Create the Gym Env
    env = gym.make(params['env_name'])

    dream_vis(env, mdnrnn, vae, params, mdn_hps, "dream_1")
    dream_vis(env, mdnrnn, vae, params, mdn_hps, "dream_2")
    dream_vis(env, mdnrnn, vae, params, mdn_hps, "dream_3")

    frame_vis(env, mdnrnn, vae, params, mdn_hps, "cmp_1")
    frame_vis(env, mdnrnn, vae, params, mdn_hps, "cmp_2")


if __name__ == "__main__":
    main()