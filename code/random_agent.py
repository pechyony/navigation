import numpy as np
import random

class RandomAgent(Agent):
    """Interacts with environment using random actions"""

    def __init__(self, action_size, seed):
        """Initializes a RandomAgent object.
        
        Params
        ======
            action_size (int): number of possible actions
            seed (int): random seed
        """
        super().__init__()
        self.action_size = action_size
        self.seed = random.seed(seed)

    def step(self, state, action, reward, next_state, done):
        """Learns from interaction with environment.
        
        Params
        ======
            state (array_like): current state
            action (int): action taken at the current state 
            reward (float): reward from an action
            next_state (array_like): next state of environment
            done (boolean): true if the next state is the final one, false otherwise
        """
        pass

    def act(self, state, eps=0.):
        """Finds the best action for a given state.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            
        Returns
        =======
            action (int): action that should be taken at the current state
        """
        return random.choice(np.arange(self.action_size)) 
        
    


