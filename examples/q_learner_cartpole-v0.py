# -*- coding: utf-8 -*-

__author__    = "Christian Richter"
__copyright__ = "Copyright 2019, TU Dresden"
__license__   = "GPL"
__credits__   = ["Christian Richter"]
__email__     = "christian.richter1@tu-dresden.de"
__project__   = "FmiRL"
__version__   = "0.1.0"

"""
Original code: https://gist.github.com/carlos-aguayo/3df32b1f5f39353afa58fbc29f9227a2 
Modified by: Christian Richter
"""

import pandas as pd
import numpy as np
import random

from fmirl.envs import InvertedPendulum
from fmirl.agents import QLearner



def cart_pole_with_qlearning():
    #env = gym.make('CartPole-v0')
    env = InvertedPendulum()
    
    #from gym.wrappers import Monitor
    #experiment_filename = './cartpole-experiment-1'
    #env = Monitor(env, experiment_filename)

    goal_average_steps = 195

    number_of_features = env.observation_space.shape[0]
    last_time_steps = np.ndarray(0)

    cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
    pole_angle_bins = pd.cut([-2, 2], bins=10, retbins=True)[1][1:-1]
    cart_velocity_bins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
    angle_rate_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]

    def build_state(features):
        return int("".join(map(lambda feature: str(int(feature)), features)))

    def to_bin(value, bins):
        return np.digitize(x=[value], bins=bins)[0]

    learner = QLearner(num_states=15 ** number_of_features,
                       num_actions=env.action_space.n,
                       alpha=0.2,
                       gamma=1,
                       random_action_rate=0.5,
                       random_action_decay_rate=0.99)

    for episode in range(50000):
        observation = env.reset()
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
        state = build_state([to_bin(cart_position, cart_position_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(cart_velocity, cart_velocity_bins),
                             to_bin(angle_rate_of_change, angle_rate_bins)])
        action = learner.set_initial_state(state)
        steps = 0

        while True:
            steps += 1
            
            observation, reward, done, info = env.step(action)

            cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

            state_prime = build_state([to_bin(cart_position, cart_position_bins),
                                       to_bin(pole_angle, pole_angle_bins),
                                       to_bin(cart_velocity, cart_velocity_bins),
                                       to_bin(angle_rate_of_change, angle_rate_bins)])

            if done:
                reward = -200            

            action = learner.move(state_prime, reward)
            env.render()

            if done:
                print("Steps: ", steps)
                break

        if last_time_steps.mean() > goal_average_steps:
            print("Goal reached!")
            print("Episodes before solve: ", episode + 1)
            break



if __name__ == "__main__":
    random.seed(0)
    cart_pole_with_qlearning()