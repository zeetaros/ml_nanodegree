import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon=None, alpha=None, gamma=None, sarsa=None):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - sarsa: determines which SARSA algorithm to use; take value from 0, "max" or "expected" 
                 corresponding to SARSA(0), Q-Learning & Expected SARSA respectively
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = epsilon or 0.005
        self.gamma = gamma or 1
        self.alpha = alpha or 0.01
        self.sarsa = sarsa or 0

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        probs = self.get_policy(state)
        return np.random.choice(self.nA, p=probs)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        if self.sarsa == 0:
            # get action in next time step: At+1
            next_action = self.select_action(next_state)
            self.Q[state][action] = self.update_Q(self.Q[state][action], self.Q[next_state][next_action], reward)
        elif self.sarsa == 'max':
            self.Q[state][action] = self.update_Q(self.Q[state][action], np.argmax(self.Q[next_state]), reward)
        elif self.sarsa == 'expected':
            self.Q[state][action] = self.update_Q(self.Q[state][action], np.dot(self.get_policy(next_state), self.Q[next_state]), reward)

        # Imporvement: create a function to update Q
        
    def get_policy(self, state):
        probs = np.array([self.eps / self.nA] * self.nA)
        probs[np.argmax(self.Q[state])] = 1 - self.eps + self.eps / self.nA
        return probs

    def update_Q(self, Qsa, next_Q, reward):
        return Qsa + self.alpha * (reward + self.gamma * next_Q - Qsa)
