import json
import gym
import controller_model
import sys
sys.path.append('../vae-cnn/')
from vae import VAE
sys.path.append('../mdn-rnn/')
from mdn import MDNRNN

''' JSON path
    p['env_name']
    p['num_eps']
    p['num_steps']
    p['seq_len']
    p['img_size']
    p['latent_size']
    p['hidden_size']
    p['exp_id']
    p['vae_weights_path']
    p['mdnrnn_weights_path']
    p['mdnrnn_hps']
    p['controller_weights_dir']
'''

class Simulation:
    def __init__(self, path): 
        self.p = json.load(open(path))
        self.load_model()

    def load_model(self, controller_weights=None):
        self.vae = VAE()
        self.mdn_rnn = MDNRNN(self.p['mdnrnn_hps'])
        self.controller = ControllerModel()
        self.vae.load_model(self.p['vae_weights_path'])
        self.mdn_rnn.restore(self.p['mdnrnn_weights_path'])
        if controller_weights:
            self.controller.load_weights(self.p['controller_weights_path'])

    def simulate(self, render=False):
        env = gym.make(self.p['env_name'])
        rewards = []
        for i in range(self.p['num_eps']):
            obs = env.reset()
    
            # initialize hidden + action variables
            self.mdn_rnn.reset_states()
            a = env.action_space.sample()
            h = np.zeros(1, self.p['hidden_size'])
            c = np.zeros(1, self.p['hidden_size'])

            total_reward = 0
            for t in range(self.p['num_steps']):
                img = env.render(mode='rgb_array')
        
                # compute action
                z = self.vae.encode_image(img)
                h, c = self.mdn_rnn.rnn_next_state(z, a, h, c)
                a = self.controller.get_action(np.vstack((z, h)))

                obs, reward, done, info = env.step(a)
                total_reward += reward
                if done:
                    print('Episode finished after {} timesteps'.format(t+1))
                    break
            rewards.append(total_reward)
        env.close()
        return -np.mean(rewards)
