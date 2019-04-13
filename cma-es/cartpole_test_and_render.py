import sys
sys.path.append('../model')
from model import Model
import gym
import cma
import numpy as np
import time

model = Model([4, 1])
env = gym.make('CartPole-v0')

def simulate(params, render=False, stop=True):
    model.load_weights(params)
    obs = env.reset()
    tot_reward = 0
    for _ in range(1000):
        if render:
            env.render()
            time.sleep(0.05)
        action = model.get_action(obs)
        obs, reward, done, info = env.step(int(round(action[0])))
        tot_reward += reward
        if done and stop:
            break
    return -tot_reward

es = cma.CMAEvolutionStrategy(model.num_params * [0], 0.5)
i = 0
while not es.stop():
    solutions = es.ask()
    scores = [simulate(x) for x in solutions]
    if i == 30:
        ind = np.argmax(scores)
        simulate(solutions[ind], render=True, stop=False)
        break
    es.tell(solutions, scores)
    # es.logger.add()  # write data to disc to be plotted
    es.disp()
    i += 1
# es.result_pretty()
# cma.plot()  # shortcut for es.logger.plot()

env.close()
