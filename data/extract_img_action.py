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
parser.add_argument('--size', type=int, default=60, help='width/height of image')

def compress_image(img, size=60):
	return resize(img, (size, size), mode='reflect', anti_aliasing=True)

if __name__ == "__main__":
	args = parser.parse_args()
	env = gym.make(args.env_name)
	ep_img_arr = []
	ep_act_arr = []
	for i_episode in range(args.episodes):
		observation = env.reset()
		step_img_arr = []
		step_act_arr = []
		for t in range(args.steps):
			if args.box2d:
				image = observation
			else:
				image = env.render(mode='rgb_array')
			image = compress_image(image, size=args.size)
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			if done:
				break
			step_img_arr.append(image)
			step_act_arr.append(action)
	            
		ep_img_arr.append(step_img_arr)
		ep_act_arr.append(step_act_arr)

	np.savez_compressed(args.env_name+"_img_"+str(args.episodes)+"_"+str(args.steps), *ep_img_arr)
	np.savez_compressed(args.env_name+"_act_"+str(args.episodes)+"_"+str(args.steps), *ep_act_arr)




		
