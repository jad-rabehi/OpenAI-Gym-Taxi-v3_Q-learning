from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)

print("alpha=" + str(agent.alpha))
print("gamma=" + str(agent.gamma))
print("epsilon=" + str(agent.epsilon))