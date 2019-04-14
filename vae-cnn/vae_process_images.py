from vae import VAE
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import argparse

def vae_process_images(images, weights, dimension, latents, encode, decode):
    conv_vae = VAE()
    conv_vae.make_vae(images + ".npz", dimension)
    conv_vae.load_model(weights + ".h5")

    latent_vectors = []
    raw_data = np.load(images + ".npz")
    if encode: 
        for f in sorted(raw_data.files):
            file_images = raw_data[f]
            latent_vectors.append([conv_vae.encode_image(np.array([image]))[0] for image in file_images])
        np.savez_compressed(images + "_latent.npz", *latent_vectors)
    
    if decode:
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
    help_ = "Name of images file without file extension"
    parser.add_argument("images", help=help_, type=str)
    help_ = "Name of weights file without file extension"
    parser.add_argument("weights", help=help_, type=str)
    help_ = "Name of latents file"
    parser.add_argument("dimension", help=help_, type=int)
    help_ = "Output encoded latent vectors in _latent.npz file"
    parser.add_argument("-l", "--latents", help=help_, type=str)
    help_ = "Dimension of latent vectors"
    parser.add_argument("-e", "--encode", help=help_, action='store_true')
    help_ = "Output reconstructed images in _recon.npz file (if only this flag is on, then assume _latent.npz inputs)"
    parser.add_argument("-d", "--decode", help=help_, action='store_true')
    args = parser.parse_args()

    vae_process_images(args.images, args.weights, args.dimension, args.latents, args.encode, args.decode)


   