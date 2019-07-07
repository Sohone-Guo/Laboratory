
import numpy as np

class Agent(object):

    def __init__(self):
        self.table = {}
        self.pre_env = [0,0]
        self.table["{}".format(self.pre_env)] = np.random.random_sample(3)

    
    def _forward(self, env):
        # prediction
        if "{}".format(env) in self.table:
            act = self.table["{}".format(env)].argmax()
        else:
            self.table["{}".format(env)] = np.random.random_sample(3)
            act = self.table["{}".format(env)].argmax()
        return act 
               

    def _backward(self, env, reward):
        argmax_pre_env = self.table["{}".format(self.pre_env)].argmax()
        argmax_env = self.table["{}".format(env)].argmax()
        self.table["{}".format(self.pre_env)][argmax_pre_env] += 0.01*reward*(self.table["{}".format(env)][argmax_env] 
                                                                - self.table["{}".format(self.pre_env)][argmax_pre_env])


    def _action(self, env, reward):
        
        # prediction
        act = self._forward(env)

        # backward
        self._backward(env, reward)

        # set pre env
        self.pre_env = env

        return act