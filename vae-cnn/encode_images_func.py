from vae import VAE
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import argparse

def encode(_input, weights, latent, reconstruct):
    conv_vae = VAE()
    conv_vae.make_vae(_input + ".npz", int(latent))
    conv_vae.load_model(weights)

    latent_vectors = []
    # for i in range(len(conv_vae.x_test)):
    #     latent_vectors.append(conv_vae.encode_image(conv_vae.x_test[i:i+1]))
    raw_data = np.load(_input + ".npz")
    for f in sorted(raw_data.files):
        images = raw_data[f]
        latent_vectors.append([conv_vae.encode_image(np.array([image]))[0] for image in images])

    np.savez_compressed(_input + "_latent.npz", *latent_vectors)
    if reconstruct:
        data = np.load(_input + "_latent.npz")
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
        np.savez_compressed(_input + "_recon.npz", *reconstructed_images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Name of input without npzs"
    parser.add_argument("-i", "--input", help=help_)
    help_ = "Name of weights file"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Dim of latent vec"
    parser.add_argument("-l", "--latent", help=help_)
    help_ = "Output reconstructed images as well"
    parser.add_argument("-r", "--reconstruct", help=help_, action='store_true')

    args = parser.parse_args()
    encode(args.input, args.weights, args.latent, args.reconstruct)
