import sys
import numpy as np
import cma
import argparse
import os

sys.path.append('../model')
from simulate import Simulation

parser = argparse.ArgumentParser(description='Extract data, train VAE, train MDN.')
parser.add_argument('json_path', type=str, help='Path to JSON file with model params.')
parser.add_argument('--dreaming', default=False, help='Whether models dreams or not')

def train_controller(json_path, dreaming=False):
    s = Simulation(json_path)

    es = cma.CMAEvolutionStrategy(s.controller.num_params * [0], 0.5)
    n_iters = 0
    rewards = []
    while not es.stop() or n_iters >= s.params['cma-es_hps']['max_iters']:
        # TODO: Number of models to make to be a param in the json
        solutions = es.ask()
        loss = []
        for x in solutions:
            s.controller.set_weights(x)
            loss.append(s.simulate(dreaming=dreaming))
        es.tell(solutions, loss)

        reward = -sum(loss)
        rewards.append(reward)
        best_sol = solutions[np.argmin(loss)]

        # es.logger.add()

        if n_iters % 100 == 0:
            # TODO: Better naming
            np.save('../' + s.params['cma-es_hps']['weights_dir'] +
                '/cma_model_rewards_{}'.format(n_iters), np.array(rewards))
            np.save('../' + s.params['cma-es_hps']['weights_dir'] +
                '/cma_model_{}'.format(n_iters), np.array(best_sol))
        n_iters += 1

if __name__ == '__main__':
    args = parser.parse_args()
    train_controller(args.json_path, dreaming=args.dreaming)
