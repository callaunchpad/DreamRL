import gym
import numpy as np
from numpy import load
import argparse

parser = argparse.ArgumentParser(description='Generate nested array of images given an environment.')

parser.add_argument('env_name', type=str, help='environment name')
parser.add_argument('--episodes', type=int, default=100, help='number of episodes')
parser.add_argument('--steps', type=int, default=250, help='number of steps')


# parser.add_argument('--box2d', type=bool, help='whether environment is box2d or not')

if __name__ == "__main__":
	args = parser.parse_args()
	env = gym.make(args.env_name)
	ep_img_arr = []
	ep_act_arr = []
	for i_episode in range(args.episodes): #100
		observation = env.reset()
		step_img_arr = []
		step_act_arr = []
		for t in range(args.steps): #250
			a = env.render(mode='rgb_array')
			a = a[::5, ::5, :]
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			# if done:
			# 	break
			step_img_arr.append(a)
			step_act_arr.append(action)
	            
		ep_img_arr.append(step_img_arr)
		ep_act_arr.append(step_act_arr)

	np.savez(args.env_name+"_image_"+str(args.episodes)+"_"+str(args.steps), *ep_img_arr)
	np.savez(args.env_name+"_action_"+str(args.episodes)+"_"+str(args.steps), *ep_act_arr)




		
