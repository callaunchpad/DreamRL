from vae import VAE
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    conv_vae = VAE()
    # make this in to argument parser
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
    conv_vae.make_vae(args.input + ".npz", int(args.latent))
    conv_vae.load_model(args.weights)

    # columns = 2
    # rows = 3
    # fig=plt.figure(figsize=(8, 8))
    # t=1
    # for i in range(1, columns*rows +1):
    #     if (i % 2 == 1):
    #         img = conv_vae.x_test[t-1:t]*255
    #         t += 1
    #     else:
    #         img = conv_vae.decode_latent(conv_vae.encode_image(img))
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(img[0])
    # plt.show()

    latent_vectors = []
    # for i in range(len(conv_vae.x_test)):
    #     latent_vectors.append(conv_vae.encode_image(conv_vae.x_test[i:i+1]))
    raw_data = np.load(args.input + ".npz")
    for f in sorted(raw_data.files):
        images = raw_data[f]
        latent_vectors.append([conv_vae.encode_image(np.array([image]))[0] for image in images])

    np.savez_compressed(args.input + "_latent.npz", *latent_vectors)
    if args.reconstruct:
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
