'''
Created on 8 Mar 2023

@author: ucacsjj
'''

import numpy as np

from .td_controller import TDController

# Simplified version of the predictor from S+B

class SARSA(TDController):
    '''
    classdocs
    '''

    def __init__(self, environment):
        TDController.__init__(self, environment)

    def initialize(self):
        
        TDController.initialize(self)
        
        self._v.set_name("SARSA Expected Value Function")
        self._pi.set_name("SARSA Greedy Policy")
                    
    def _update_action_and_value_functions_from_episode(self, episode):
        
        # Q2g:
        # Complete implementation of this method
        # Each time you update the state value function, you will need to make a
        # call of the form:
        #
        # self._update_q_and_policy(coords, a, new_q) 
        #
        # This calls a method in the TDController which will update the
        # Q value estimate in the base class and will update
        # the greedy policy and estimated state value function
        
        for i in range(episode._number_of_steps - 1):

            # get the state, action and reward for the current step of the episode
            current_state = episode.state(i)
            current_action = episode.action(i)
            current_reward = episode.reward(i)
            next_state = episode.state(i+1)
            next_action = episode.state(i+1)

            # get the coords of the current state and the old value of Q at the current state
            xy = current_state.coords()
            old_Q = self._Q[xy[0], xy[1], current_action]

            # get the coords of the next state and the value of Q at the next state 
            next_xy = next_state.coords()
            next_Q = self._Q[next_xy[0], next_xy[1], next_action]

            # calculate the new value of Q 
            error = current_reward + self.gamma() * next_Q - old_Q
            new_Q = old_Q + self.alpha() * error

            self._update_q_and_policy(xy, current_action, new_Q) 
