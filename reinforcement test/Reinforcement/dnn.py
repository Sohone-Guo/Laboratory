
from keras.models import Sequential
from keras.layers import Dense
import keras

import numpy as np

class Agent(object):

    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(units=16, activation='tanh', input_dim=2))
        self.model.add(Dense(units=32, activation='tanh'))
        self.model.add(Dense(units=3, activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True),metrics=['accuracy'])

        self.pre_env = [0,0]


    def _forward(self, env):
        # prediction
        pre = self.model.predict(np.array([env]))
        return pre.argmax()
               

    def _backward(self, env, reward):
        pre_ = self.model.predict(np.array([self.pre_env]))
        now_ = self.model.predict(np.array([env]))
        
        self.model.fit(np.array([self.pre_env]), reward*now_, epochs=1, batch_size=1)


    def _action(self, env, reward):
        
        # prediction
        act = self._forward(env)

        # backward
        self._backward(env, reward)

        # set pre env
        self.pre_env = env

        return act