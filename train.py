import os
import numpy as np
import sys
import json
import argparse
import gym

sys.path.insert(0, 'data')
from extract_img_action import extract, get_path_names
from action_utils import ActionUtils
sys.path.insert(0, 'vae-cnn')
from vae_process_images import vae_process_images
from vae import VAE
sys.path.insert(0, 'mdn-rnn')
from mdn import MDNRNN

parser = argparse.ArgumentParser(description='Extract data, train VAE, train MDN.')
parser.add_argument('json_path', type=str, help='Path to JSON file with model params.')
parser.add_argument('-d', '--use-previous-dataset', default=False, help='Use previously generated dataset', action='store_true')
parser.add_argument('-v', '--use-trained-vae', default=False, help='Use previously trained VAE', action='store_true')

def train(json_path):
    # TODO: suppress VAE loading print statements
    params = json.load(open(json_path))[0]

    print("Extracting data...")
    # TODO: save data across more files? and all in some specific folder too
    if not args.use_previous_dataset:
	    img_path_name, action_path_name = extract(
	        params['env_name'], params['num_eps'], params['max_seq_len'], False,
	        params['img_size'], path=params['dataset_path'])
    else:
    	print("Using previously trained dataset.")
    	img_path_name, action_path_name = get_path_names(params['dataset_path'],
    		params['env_name'],params['num_eps'], params['max_seq_len'])
    	if not os.path.isfile(img_path_name + ".npz") or not os.path.isfile(action_path_name + ".npz"):
    		return print("ERROR: One or more of the previously trained dataset paths \
    			(`{}` or `{}`) does not exist".format(img_path_name, action_path_name))

    print("Training VAE...")
    convVae = VAE()
    sys.stdout = open(os.devnull, 'w')
    convVae.make_vae(img_path_name + ".npz", params['latent_size'])
    sys.stdout = sys.__stdout__
    vae_path = params['vae_hps']['weights_path']
    if args.use_trained_vae:
    	if not os.path.isfile(vae_path):
    		return print("ERROR: No file exists at the VAE model path you passed (`{}`)".format(vae_path))
    	else:
    		print("Loading VAE model from given path.")
    	convVae.load_model(vae_path)
    else:
	    convVae.model_name = vae_path
	    convVae.epochs = params['vae_hps']['epochs']
	    convVae.train_vae()

    vae_process_images(img_path_name, vae_path, params['latent_size'], decode=False, image_size=params['img_size'])
    latent_path_name = img_path_name + '_latent.npz'

    latent = np.load(latent_path_name)
    act = np.load(action_path_name + '.npz')

    combined_input = []
    combined_output = []

    utils = ActionUtils(params['env_name'])
    action_size = utils.action_size()

    print("Saving output...")
    # TODO: Save in batches?
    for f in latent.files:
        c = np.concatenate([latent[f], np.array([utils.action_to_input(a) for a in act[f]])], axis=1)
        missing = params['max_seq_len'] + 1 - c.shape[0]
        c = np.concatenate([c, np.zeros((missing, params['latent_size'] + action_size))], axis=0)
        combined_input.append(c[:-1])
        combined_output.append(c[1:, :-action_size])

    np.save('LunarLander_MDN_in', combined_input)
    np.save('LunarLander_MDN_out', combined_output)

    # MDN Parameters
    # TODO: Change MDN to just take in entire params dictionary
    print("Configuring MDN...")
    mdn_hps = params['mdn_hps']
    mdn_hps['max_seq_len'] = params['max_seq_len']
    mdn_hps['in_width'] = params['latent_size'] + action_size
    mdn_hps['out_width'] = params['latent_size']
    mdn_hps['action_size'] = action_size
    mdn_hps['rnn_size'] = params['hidden_size']

    mdnrnn = MDNRNN(mdn_hps)
    print("Finished building MDN, starting training...")

    mdnrnn.train(np.array(combined_input), np.array(combined_output))
    print("Finished training MDN.")
    mdnrnn.save(params['mdn_hps']['weights_path'])

if __name__ == '__main__':
    args = parser.parse_args()
    train(args.json_path)
