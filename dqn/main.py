import gym
from dqn import DQNAgent

# Initialize the environment
env = gym.make('CartPole-v1')

# Initialize agent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)

# Train the agent
num_episodes = 5000
agent.train(env, num_episodes)
