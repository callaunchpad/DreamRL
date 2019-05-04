from vae import VAE
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import argparse
import os
import sys

def vae_process_images(images, weights, latent_size, latents=None, encode=True, decode=True, image_size=None):
    conv_vae = VAE()
    sys.stdout = open(os.devnull, 'w')
    if image_size:
        conv_vae.make_vae_shape(image_size, image_size, latent_size)
    else:
        conv_vae.make_vae(images + ".npz", latent_size)
    conv_vae.load_model(weights)
    sys.stdout = sys.__stdout__

    latent_vectors = []
    raw_data = np.load(images + ".npz")
    if encode:
        print("Encoding images...")
        for i in range(len(raw_data.files)):
            f = 'arr_' + str(i)
            file_images = raw_data[f]
            latent_vectors.append([conv_vae.encode_image(np.array([image]))[0] for image in file_images])
        np.savez_compressed(images + "_latent.npz", *latent_vectors)
    
    if decode:
        print("Decoding latent vectors...")
        data = ""  
        if latents: 
            data = np.load(latents + ".npz") 
        else: 
            data = np.load(images + "_latent.npz")
        # files = data.files
        # vectors = np.array(data[files[0]])
    
        reconstructed_images = []
        for f in sorted(data.files):
            latents = data[f]
            # a = a.reshape(-1, a.shape[-2], a.shape[-1])
            reconstructed_images.append([conv_vae.decode_latent(np.array([l]))[0] for l in latents])
        # print(np.shape(np.array(reconstructed_images)))

        # plt.imshow(reconstructed_images[1])

        # plt.show()
        np.savez_compressed(images + "_recon.npz", *reconstructed_images)



if __name__ == '__main__':
    # make this in to argument parser
    parser = argparse.ArgumentParser()
    help_ = "Name of images file without .npz file extension"
    parser.add_argument("images", help=help_, type=str)
    help_ = "Name of weights file with file extension"
    parser.add_argument("weights", help=help_, type=str)
    help_ = "Dimension of latent vectors"
    parser.add_argument("latent_size", help=help_, type=int)
    help_ = "Name of latents file without .npz file extension"
    parser.add_argument("-l", "--latents", help=help_, type=str)
    help_ = "Output encoded latent vectors in _latent.npz file"
    parser.add_argument("-e", "--encode", help=help_, action='store_true')
    help_ = "Output reconstructed images in _recon.npz file (if only this flag is on, then assume _latent.npz inputs)"
    parser.add_argument("-d", "--decode", help=help_, action='store_true')
    help_ = "Image size"
    parser.add_argument("-i", "--image_size", default=None, help=help_, type=int)
    args = parser.parse_args()

    vae_process_images(args.images, args.weights, args.latent_size, 
        latents=args.latents, encode=args.encode, decode=args.decode, image_size=args.image_size)


   