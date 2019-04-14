import numpy as np
import sys
sys.path.insert(0, 'data')
from extract_img_action import extract
sys.path.insert(0, 'vae-cnn')
from encode_images_func import encode


img_path_name, action_path_name = extract("LunarLander-v2", 2500, 150, False, 80)

encode(img_path_name, 'vae-cnn/LunarLander_64.h5', 64, False)
latent_path_name = img_path_name + '_latent.npz'

latent = np.load(latent_path_name) # (2500, ?, 64)
act = np.load(action_path_name + '.npz') # (2500, ?, 1)

combined_input = []
combined_output = []

def hot(tot, i):
    v = np.zeros(tot)
    v[i] = 1
    return v

for f in latent.files:
    c = np.concatenate([latent[f], np.array([hot(4, i) for i in act[f]])], axis=1)
    missing = 151 - c.shape[0]
    c = np.concatenate([c, np.zeros((missing, 68))], axis=0)
    combined_input.append(c[:-1])
    combined_output.append(c[1:, :-4])

np.save('LunarLander_MDN_in', combined_input)
np.save('LunarLander_MDN_out', combined_output)
