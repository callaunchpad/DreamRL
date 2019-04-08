import numpy as np
from scipy.special import expit

class Model:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_params = self.compute_num_params()

    def load_weights(self, weights):
        assert(len(weights) == self.compute_num_params())
        self.weights = {}
        for i in range(len(self.layer_sizes)-1):
            # weight matrices
            r, c = self.layer_sizes[i+1], self.layer_sizes[i]
            key = 'W' + str(i)
            self.weights[key] = np.array(weights[:r*c]).reshape(r,c)
            weights = weights[r*c:]
            
            # bias vectors
            key = 'b' + str(i)
            self.weights[key] = np.array(weights[:r])
            weights = weights[r:]
    
    def get_action(self, obs):
        k = len(self.layer_sizes)-1
        out = obs
        for i in range(k):
            W = self.weights['W' + str(i)]
            b = self.weights['b' + str(i)]
            score = np.dot(W, out) + b
            # enforce output between 0 and 1 (for CartPole)
            out = np.tanh(score) if i < k-1 else expit(score)
        return out

    def compute_num_params(self):
        num = 0
        for i in range(len(self.layer_sizes)-1):
            r, c = self.layer_sizes[i+1], self.layer_sizes[i]
            num += r*c + r
        return num
    
def main():
    m = Model([1, 2, 2, 1])
    m.load_weights(np.random.randn(13, 1).tolist())
    out = m.get_action(np.array([1]))
    print(out)

if __name__ == '__main__':
    main()
