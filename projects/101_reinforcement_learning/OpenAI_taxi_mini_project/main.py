from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
agent = Agent(epsilon= 0.001, alpha=0.2, sarsa='expected')
avg_rewards, best_avg_reward = interact(env, agent)