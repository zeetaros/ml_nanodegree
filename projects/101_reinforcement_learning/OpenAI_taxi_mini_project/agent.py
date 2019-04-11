import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon=None, alpha=None, gamma=None):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = epsilon or 0.005
        self.gamma = gamma or 1
        self.alpha = alpha or 0.01
        

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done, sarsa=None):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        - sarsa: determines which SARSA algorithm to use; take value from 0, "max" or "expected" 
                 corresponding to SARSA(0), Q-Learning & Expected SARSA respectively
        """
        self.sarsa = sarsa or 0 
        self.Q[state][action] += 1

