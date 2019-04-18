import sys
import numpy as np
import cma

sys.path.append('../model')
from simulate import Simulation

# TODO: function-ify this
json_path = '../params_template.json'
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
        loss.append(s.simulate())
    es.tell(solutions, loss)
 
    reward = -sum(loss)
    rewards.append(reward)
    best_sol = solutions[np.argmin(loss)]
 
    es.logger.add()

    if n_iters % 10 == 0:
        # TODO: Better naming
        np.save('../' + s.params['cma-es_hps']['weights_dir'] +
            '/cma_model_{}'.format(n_iters), np.array(best_sol))
    n_iters += 1
