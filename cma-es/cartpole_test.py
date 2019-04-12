from sklearn.decomposition import PCA
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import gym, cma
import sys
sys.path.append('../model')
from model import Model
import time

env_name = 'CartPole-v0'
model = Model([4, 1])

def run_test():
    env = gym.make(env_name)
    es = cma.CMAEvolutionStrategy(model.num_params * [0], 0.5)
    
    num_sols = 8 # inherent to library
    rewards = []
    projs = np.array([]).reshape(0, 2)
    iters = 0
    while not es.stop():
        solutions = es.ask()
        loss = [simulate(x, env) for x in solutions]
        es.tell(solutions, loss)

        reward = -sum(loss)
        rewards.append(reward)
        best_sol = solutions[np.argmin(loss)]

        # proj weights
        mean_sol = np.mean(np.array(solutions), axis=0)
        proj = PCA(n_components=2).fit_transform(np.array(solutions)-mean_sol)
        projs = np.vstack((projs, proj))

        es.logger.add()
        # es.disp()

        if iters % 10 == 0: visualize_env(best_sol)
        iters += 1
    env.close()
    visualize_env(best_sol)
    
    def animate(i):
        ax1.clear()
        ax1.scatter(projs[:i*num_sols+1,0],projs[:i*num_sols+1,1])

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ani = animation.FuncAnimation(fig, animate, interval=10)
    ax2 = fig.add_subplot(122)
    ax2.plot(rewards)
    plt.show()

def visualize_env(params, N=200):
    model.load_weights(params)
    test_env = gym.make(env_name)
    obs = test_env.reset()
    for _ in range(N):
        time.sleep(0.03)
        test_env.render()
        action = model.get_action(obs)
        obs, reward, done, info = test_env.step(int(round(action[0])))
        if done: break
    test_env.close()

def simulate(params, env, N=200):
    model.load_weights(params)
    obs = env.reset()
    total_reward = 0
    for _ in range(N):
        action = model.get_action(obs)
        obs, reward, done, info = env.step(int(round(action[0])))
        total_reward += reward
        if done: break
    return -total_reward

if __name__ == '__main__':
    run_test()
