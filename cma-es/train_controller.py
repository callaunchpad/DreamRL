json_path = ''
s = Simulation(json_path)

es = cma.CMAEvolutionStrategy(s.controller.num_params * [0], 0.5)
n_iters = 0
while not es.stop():
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
        np.save(s.p['controller_weights_path']+'_{}'.format(n_iters), np.array(best_sol))
    n_iters += 1
