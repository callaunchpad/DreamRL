import gym
import json
import argparse
import sys
sys.path.insert(0, 'data/')
import env_to_imgarr
sys.path.insert(0, 'mdn-rnn/')
import mdn
sys.path.insert(0, 'vae-cnn')
import vae

parser = argparse.ArgumentParser()
parser.add_argument('env_name', type=str, help='environment name')
parser.add_argument('json_path', type=str, help='json file path')
parser.add_argument('reward_path', type=str, help='reward file path')

args = parser.parse_args()
path = args.json_path
env_name = args.env_name
reward_path = args.reward_path

# load models
vae = load_model()
mdn_rnn = load_model()
cmaes = load_weights()

# read experiment parameters
exp_params = json.load(open(path))
n_hidden = exp_params['n_hidden']
# n_latent = exp_params['n_latent']
n_action = exp_params['n_action']
# n_mix = exp_params['n_mix']
num_episodes = exp_params['num_episodes']
num_steps = exp_params['num_steps']

rewards = []

env = gym.make(env_name)
for i_episode in range(num_episodes):
    obs = env.reset()
    
    # initialize hidden and action variables
    h = np.zeros(n_hidden)
    a = np.zeros(n_action)
    
    cum_reward = 0
    for t in range(num_steps):
        img = env.render(mode='rgb_array')
        
        # compute action
        z = vae.encode(img)
        z, h = mdn_rnn.get_next_z_h(z, h, a)
        a = cmaes.evaluate(z, h)

        obs, reward, done, info = env.step(a)
        cum_reward += reward
        if done:
            print('Episode finished after {} timesteps'.format(t+1))
            break
    rewards.append(reward)

np.save(reward_path, np.array(rewards))
