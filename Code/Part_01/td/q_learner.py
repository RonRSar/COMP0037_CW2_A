'''
Created on 8 Mar 2023

@author: steam
'''

import random
import numpy as np

from .td_controller import TDController

# Simplified version of the predictor from S+B

class QLearner(TDController):
    '''
    classdocs
    '''

    def __init__(self, environment):
        TDController.__init__(self, environment)

    def initialize(self):
               
        # Set up experience replay buffer
        TDController.initialize(self)
        
        # Change names to change titles on drawn windows
        self._v.set_name("Q-Learning Expected Value Function")
        self._pi.set_name("Q-Learning Greedy Policy")
            
    def _update_action_and_value_functions_from_episode(self, episode):
        
        # Q2b:
        # Complete implementation of this method
        # Each time you update the state value function, you will need to make a
        # call of the form:
        #
        # self._update_q_and_policy(coords, a, new_q) 
        #
        # This calls a method in the TDController which will update the
        # Q value estimate in the base class and will update
        # the greedy policy and estimated state value function

        # Q-learning update rule:
        # Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

        for i in range(episode._number_of_steps):

            s = episode.state(i) # current state
            a = episode.action(i)  # current action
            r = episode.reward(i)  # current reward
            next_s = episode.state(i+1) # next state

            xy = s.coords()

            # check for terminal state
            if next_s == None:
                Q_max = 0
            else:
                next_xy = next_s.coords()
                # get the action that maximises the reward from the next state
                q_vals = self._Q[next_xy[0], next_xy[1], :]
                a_max = np.argmax(q_vals)
                Q_max = self._Q[next_xy[0], next_xy[1], a_max]

            # Q-value with max Q-value of next state-action pair
            q_target = r + self._gamma *  Q_max
                                                     
            # current Q-value for the current state-action pair
            q_current = self._Q[xy[0], xy[1], a]

            # Update Q-value using learning rate and Q-learning update rule
            new_q = q_current + self.alpha() * (q_target - q_current)
            self._update_q_and_policy(xy, a, new_q)  
