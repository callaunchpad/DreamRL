import gym
import numpy as np
from numpy import load
import argparse
from skimage.transform import resize

parser = argparse.ArgumentParser(description='Generate nested array of images given an environment.')

parser.add_argument('env_name', type=str, help='environment name')
parser.add_argument('--episodes', type=int, default=100, help='number of episodes')
parser.add_argument('--steps', type=int, default=250, help='maximum sequence per episode')
parser.add_argument('--box2d', type=bool, default=False, help='whether environment is box2d or not')
parser.add_argument('--size', type=int, default=120, help='width/height of image')

def compress_image(img, size=60):
	return resize(img, (size, size), mode='reflect', anti_aliasing=True)

def get_path_names(path, env_name, episodes, steps):
	img_path_name = path if path else env_name
	act_path_name = path if path else env_name
	img_path_name += "_img_" + str(episodes) + "_" + str(steps)
	act_path_name += "_act_" + str(episodes) + "_" + str(steps)
	return img_path_name, act_path_name

def extract(env_name, episodes, steps, box2d, size, path=None):
	env = gym.make(env_name)
	ep_img_arr = []
	ep_act_arr = []
	for i_episode in range(episodes):
		observation = env.reset()
		step_img_arr = []
		step_act_arr = []
		for t in range(steps):
			if box2d:
				image = observation
			else:
				image = env.render(mode='rgb_array')
			image = compress_image(image, size=size)
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			if done:
				break
			step_img_arr.append(image)
			step_act_arr.append(action)

		ep_img_arr.append(step_img_arr)
		ep_act_arr.append(step_act_arr)
	img_path_name, act_path_name = get_path_names(path, env_name, episodes, steps)
	np.savez_compressed(img_path_name, *ep_img_arr)
	np.savez_compressed(act_path_name, *ep_act_arr)
	return img_path_name, act_path_name

if __name__ == "__main__":
	args = parser.parse_args()
	extract(args.env_name, args.episodes, args.steps, args.box2d, args.size)




		
