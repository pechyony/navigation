from abc import ABCMeta, abstractmethod

class Agent(metaclass=ABCMeta):
    """Abstract class of an agent that interacts with and learns from environment."""

    def __init__(self):
        """Initializes an agent."""
        pass

    @abstractmethod
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

    @abstractmethod
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
        pass
        
    


