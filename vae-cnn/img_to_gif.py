import numpy as np
import imageio
import argparse

parser = argparse.ArgumentParser(description='Generate a gif per sequence given the npz file.')

parser.add_argument('npz_file_name', type=str, help='npz file name')

if __name__ == "__main__":
	args = parser.parse_args()
	img_arr = np.load(args.npz_file_name)

	def convert_to_int(arr):
	    return (arr * 255).astype(int)

	for f in sorted(img_arr.files[:10]):
		seq = convert_to_int(img_arr[f])
		imageio.mimsave(args.npz_file_name+'_seq_'+f[-1]+'.gif', seq)


