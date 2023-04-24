'''
Created on 8 Mar 2023

@author: ucacsjj
'''

from monte_carlo.episode_sampler import EpisodeSampler

from .td_algorithm_base import TDAlgorithmBase

class TDPolicyPredictor(TDAlgorithmBase):

    def __init__(self, environment):
        
        TDAlgorithmBase.__init__(self, environment)
        
        self._minibatch_buffer= [None]
                
    def set_target_policy(self, policy):        
        self._pi = policy        
        self.initialize()
        self._v.set_name("TDPolicyPredictor")
        
    def evaluate(self):
        
        episode_sampler = EpisodeSampler(self._environment)
        
        for episode in range(self._number_of_episodes):

            # Choose the start for the episode            
            start_x, start_a  = self._select_episode_start()
            self._environment.reset(start_x) 
            
            # Now sample it
            new_episode = episode_sampler.sample_episode(self._pi, start_x, start_a)

            # If we didn't terminate, skip this episode
            if new_episode.terminated_successfully() is False:
                continue
            
            # Update with the current episode
            self._update_value_function_from_episode(new_episode)
            
            # Pick several randomly from the experience replay buffer and update with those as well
            for _ in range(min(self._replays_per_update, self._stored_experiences)):
                episode = self._draw_random_episode_from_experience_replay_buffer()
                self._update_value_function_from_episode(episode)
                
            self._add_episode_to_experience_replay_buffer(new_episode)
            
    def _update_value_function_from_episode(self, episode):

        # Q1e:
        # Complete implementation of this method
        # Each time you update the state value function, you will need to make a
        # call of the form:
        #
        # self._v.set_value(x_cell_coord, y_cell_coord, new_v)

        for i in range(episode._number_of_steps):

            # get the state, action and reward for the current step of the episode
            current_state = episode.state(i)
            current_reward = episode.reward(i)
            next_state = episode.state(i+1)

            # get the coords of the current state and the old value of v at the current state
            xy = current_state.coords()
            old_v = self._v.value(xy[0], xy[1])

            # get the coords of the next state and the value of v at the next state
            next_xy = next_state.coords()
            next_v = self._v.value(next_xy[0], next_xy[1])

            # calculate the new value of v  
            error = current_reward + self.gamma() * next_v - old_v
            new_v = old_v + self.alpha() * error

            self._v.set_value(xy[0], xy[1], new_v)
