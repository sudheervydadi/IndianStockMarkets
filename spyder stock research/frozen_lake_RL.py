# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:33:13 2020

@author: sudhe
"""

import gym
import numpy as np

env = gym.make('FrozenLake-v0')

print(env.observation_space.n)

value_table = np.zeros(env.observation_space.n)
no_of_iterations = 100000