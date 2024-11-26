import gym
from a2c import A2C

env = gym.make('LunarLander-v2')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = A2C(state_dim, action_dim, learning_rate=0.001, gamma=0.99)

agent.train(env, num_episodes=1000)
