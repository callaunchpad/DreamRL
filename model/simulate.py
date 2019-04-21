import json
import gym
import numpy as np
from controller_model import ControllerModel
import sys
sys.path.append('../data/')
from extract_img_action import compress_image
from action_utils import ActionUtils
sys.path.append('../vae-cnn/')
from vae import VAE
sys.path.append('../mdn-rnn/')
from mdn import MDNRNN

class Simulation:
    def __init__(self, path):
        self.params = json.load(open(path))[0]
        self.load_model()
        self.env = gym.make(self.params['env_name'])

    def load_model(self, controller_weights=None):
        p = self.params
        self.action_utils = ActionUtils(p['env_name'])
        self.action_size = self.action_utils.action_size()

        self.vae = VAE()
        # TODO: Make VAE not need to load entire datset to be made
        self.vae.make_vae('../' + p['dataset_path'] + "_img_" + str(p['num_eps']) + "_" + str(p['max_seq_len']) + '.npz',
            p['latent_size'])
        self.vae.load_model('../' + p['vae_hps']['weights_path'])

        # TODO: Make MDN just take in all of params.
        mdn_hps = p['mdn_hps']
        mdn_hps['max_seq_len'] = p['max_seq_len']
        mdn_hps['in_width'] = p['latent_size'] + self.action_size
        mdn_hps['out_width'] = p['latent_size']
        mdn_hps['action_size'] = self.action_size
        mdn_hps['rnn_size'] = p['hidden_size']
        mdn_hps['batch_size'] = 1
        mdn_hps['max_seq_len'] = 1
        mdn_hps['use_recurrent_dropout'] = 0
        mdn_hps['training'] = 0
        # self.mdn_rnn = MDNRNN(mdn_hps)
        # hps_inf = MDNRNN.set_hps_to_inference(hps)
        self.mdn_rnn = MDNRNN(mdn_hps)
        self.mdn_rnn.load('../' + p['mdn_hps']['weights_path'])

        self.controller = ControllerModel([p['latent_size'] + p['hidden_size'], self.action_size])
        if controller_weights:
            self.controller.load_weights(controller_weights)

    def simulate(self, dreaming=False, render=False):
        rewards = []
        for i in range(1):
            obs = self.env.reset()

            # initialize hidden + action variables
            state = self.mdn_rnn.rnn_init_state()
            a = self.action_utils.action_to_input(self.env.action_space.sample())
            h = np.zeros((1, self.params['hidden_size']))
            c = np.zeros((1, self.params['hidden_size']))

            total_reward = 0
            if dreaming:
                img = self.env.render(mode='rgb_array')
                img = compress_image(img, size=self.params['img_size'])
                z = self.vae.encode_image(np.array([img]))[0]
                for t in range(self.params['max_seq_len']):
                    z_current = z.copy()
                    z, state = self.mdn_rnn.sample_z(z_current, a, state)
                    z = z[0][0]
                    h, c = state[0], state[1]
                    out = self.controller.get_action(np.concatenate((z_current, h[0])))

                    obs, reward, done, info = self.env.step(self.action_utils.output_to_action(out))
                    total_reward += reward
                    if done:
                        print('Episode finished after {} timesteps'.format(t+1))
                        break
            else:
                for t in range(self.params['max_seq_len']):
                    img = self.env.render(mode='rgb_array')
                    img = compress_image(img, size=self.params['img_size'])

                    # compute action
                    z = self.vae.encode_image(np.array([img]))[0]
                    state = self.mdn_rnn.rnn_next_state(z, a, state)
                    h, c = state[0], state[1]
                    out = self.controller.get_action(np.concatenate((z, h[0])))

                    a = self.action_utils.output_to_action(out)
                    obs, reward, done, info = self.env.step(a)
                    total_reward += reward
                    if done:
                        print('Episode finished after {} timesteps'.format(t+1))
                        break
            rewards.append(total_reward)
        return -np.mean(rewards)
