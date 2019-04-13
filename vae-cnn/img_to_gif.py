import numpy as np
import imageio
import argparse

parser = argparse.ArgumentParser(description='Generate a gif per sequence given the npz file.')

parser.add_argument('npz_file_name', type=str, help='npz file name')
parser.add_argument('--episodes', type=int, default=100, help='number of episodes')

if __name__ == "__main__":
	args = parser.parse_args()
	img_arr = np.load(args.npz_file_name)

	def convert_to_int(arr):
	    return (arr * 255).astype(int)

	for i in range(args.episodes):
		seq = convert_to_int(img_arr['arr_'+str(i)])
		imageio.mimsave(args.npz_file_name+'_seq_'+str(i)+'.gif', seq)


