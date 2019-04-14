from vae import VAE
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    conv_vae = VAE()
    # make this in to argument parser
    parser = argparse.ArgumentParser()
    help_ = "Name of input file without file extension"
    parser.add_argument("-i", "--input", help=help_)
    help_ = "Name of weights file without file extension"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Name of latents file"
    parser.add_argument("-l", "--latents", help=help_)
    help_ = "Dimension of latent vectors"
    parser.add_argument("-d", "--dimension", help=help_)
    help_ = "Output encoded latent vectors in _latent.npz file"
    parser.add_argument("-e", "--encode", help=help_, action='store_true')
    help_ = "Output reconstructed images in _recon.npz file (if only this flag is on, then assume _latent.npz inputs)"
    parser.add_argument("-r", "--reconstruct", help=help_, action='store_true')

    args = parser.parse_args()

    conv_vae.make_vae(args.input + ".npz", int(args.dimension))
    conv_vae.load_model(args.weights + ".h5")

    latent_vectors = []
    raw_data = np.load(args.input + ".npz")
    if args.encode: 
        for f in sorted(raw_data.files):
            images = raw_data[f]
            latent_vectors.append([conv_vae.encode_image(np.array([image]))[0] for image in images])
        np.savez_compressed(args.input + "_latent.npz", *latent_vectors)
    

    if args.reconstruct:
        data = ""  
        if args.latents: 
            data = np.load(args.latents + ".npz") 
        else: 
            data = np.load(args.input + "_latent.npz")
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
        np.savez_compressed(args.input + "_recon.npz", *reconstructed_images)
