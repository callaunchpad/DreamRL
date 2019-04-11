import sys
sys.path.append('../model')
from model import Model
import gym
import cma
import numpy as np

model = Model([4, 64, 2])
env = gym.make('CartPole-v0')

def simulate(params):
    model.load_weights(params)
    obs = env.reset()
    tot_reward = 0
    for _ in range(1000):
        action = model.get_action(obs)
        obs, reward, done = env.step(np.round_(action))
        tot_reward += reward
        if done:
            break
    return -reward

es = cma.CMAEvolutionStrategy(model.num_params * [0], 0.5)
while not es.stop():
    solutions = es.ask()
    es.tell(solutions, [simulate(x) for x in solutions])
    es.logger.add()  # write data to disc to be plotted
    es.disp()
es.result_pretty()
cma.plot()  # shortcut for es.logger.plot()

env.close()
