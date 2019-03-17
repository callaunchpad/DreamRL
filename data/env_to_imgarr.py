import gym
import numpy as np
from numpy import load
import argparse

parser = argparse.ArgumentParser(description='Generate nested array of images given an environment.')

parser.add_argument('env_name', type=str, help='environment name')
parser.add_argument('--episodes', type=int, default=100, help='number of episodes')
parser.add_argument('--steps', type=int, default=250, help='number of steps')


# parser.add_argument('--box2d', type=bool, help='whether environment is box2d or not')

def compress(img):
	return img[::5, ::5, :]

if __name__ == "__main__":
	args = parser.parse_args()
	env = gym.make(args.env_name)
	ep_arr = []
	for i_episode in range(args.episodes): #100
		observation = env.reset()
		step_arr = []
		for t in range(args.steps): #250
			img = env.render(mode='rgb_array')
			img = compress(img)
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			# if done:
			# 	break
			step_arr.append(a)
		ep_arr.append(step_arr)

	# np.savez(args.env_name+args.episodes+args.steps, *ep_arr)
	np.savez(args.env_name+"_"+str(args.episodes)+"_"+str(args.steps), *ep_arr)