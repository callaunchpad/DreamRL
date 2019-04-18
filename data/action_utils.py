import gym
import numpy as np

def one_hot(tot, i):
    v = np.zeros(tot)
    v[i] = 1
    return v

class ActionUtils():
    def __init__(self, env_name):
        self.env_name = env_name
        self.action_space = gym.make(env_name).action_space
        self.type = type(self.action_space)

    def action_size(self):
        if self.type == gym.spaces.Discrete:
            return self.action_space.n
        elif self.type == gym.spaces.Box:
            return np.product(self.action_space.shape)
    
    def action_to_input(self, action):
        if self.type == gym.spaces.Discrete:
            return one_hot(self.action_space.n, action)
        elif self.type == gym.spaces.Box:
            return np.array(action).flatten()

    def output_to_action(self, output):
        if self.type == gym.spaces.Discrete:
            return np.argmax(output)
        elif self.type == gym.spaces.Box:
            reshaped = np.reshape(output, self.action_space.shape)
            #TODO: clip at max and min values of action space
            return reshaped