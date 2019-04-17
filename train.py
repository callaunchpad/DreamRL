import numpy as np
import sys
import json
import argparse

sys.path.insert(0, 'data')
from extract_img_action import extract
sys.path.insert(0, 'vae-cnn')
from encode_images_func import encode
from vae import VAE
sys.path.insert(0, 'mdn-rnn')
from mdn import MDNRNN

parser = argparse.ArgumentParser(description='Extract data, train VAE, train MDN.')
parser.add_argument('json_path', type=str, help='Path to json with params.')
# TODO: add option to use previously trained VAE or previously extracted data

def train(json_path):
    # TODO: suppress VAE loading print statements
    params = json.load(open(json_path))[0]

    print("Extracting data...")
    img_path_name, action_path_name = extract(
        params['env_name'], params['num_eps'], params['max_seq_len'], False, params['img_size'])

    print("Training VAE...")
    convVae = VAE()
    convVae.make_vae(img_path_name + ".npz", params['latent_size'])
    vae_path = params['vae_hps']['weights_path']
    convVae.model_name = vae_path
    convVae.epochs = params['vae_hps']['epochs']
    convVae.train_vae()

    encode(img_path_name, vae_path, params['latent_size'], False)
    latent_path_name = img_path_name + '_latent.npz'

    latent = np.load(latent_path_name) # (2500, ?, 64)
    act = np.load(action_path_name + '.npz') # (2500, ?, 1)

    combined_input = []
    combined_output = []

    def hot(tot, i):
        v = np.zeros(tot)
        v[i] = 1
        return v

    print("Saving output...")
    # TODO: Distinguish between discrete/continuous action spaces here
    # TODO: Save in batches?
    for f in latent.files:
        c = np.concatenate([latent[f], np.array([hot(4, i) for i in act[f]])], axis=1)
        missing = params['max_seq_len'] + 1 - c.shape[0]
        c = np.concatenate([c, np.zeros((missing, params['latent_size'] + params['action_size']))], axis=0)
        combined_input.append(c[:-1])
        combined_output.append(c[1:, :-params['action_size']])

    np.save('LunarLander_MDN_in', combined_input)
    np.save('LunarLander_MDN_out', combined_output)

    # MDN Parameters
    # TODO: Change MDN to just take in entire params dictionary
    print("Configuring MDN...")
    mdn_hps = params['mdn_hps']
    mdn_hps['max_seq_len'] = params['max_seq_len']
    mdn_hps['in_width'] = params['latent_size'] + params['action_size']
    mdn_hps['out_width'] = params['latent_size']
    mdn_hps['action_size'] = params['action_size']
    mdn_hps['rnn_size'] = params['hidden_size']

    mdnrnn = MDNRNN(mdn_hps)
    print("Finished building MDN, starting training...")

    mdnrnn.train(np.array(combined_input), np.array(combined_output))
    print("Finished training mdn")
    mdnrnn.save(params['mdn_hps']['weights_path'])

if __name__ == '__main__':
    args = parser.parse_args()
    train(args.json_path)