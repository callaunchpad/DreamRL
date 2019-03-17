import gym
import json
import argparse
import sys
sys.path.insert(0, 'data/')
import env_to_imgarr
sys.path.insert(0, 'mdn-rnn/')
import mdn
sys.path.insert(0, 'vae-cnn')
# import relevant vae file

# TODO: finish argparse (include json_path reward_path)
parser = argparse.ArgumentParser()
parser.add_argument('env_name', type=str, help='environment name')

num_episodes = 50
num_steps = 100

# load models
# vae = load_model()
# mdn_rnn = load_model()
# cmaes = load_weights()

# read experiement parameters
exp_params = json.load(open(path))
env_name = exp_params['env_name']
n_hidden = exp_params['n_hidden']
# n_latent = exp_params['n_latent']
n_action = exp_params['n_action']
# n_mix = exp_params['n_mix']

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
