import gym
from reinforce import REINFORCE

# Create the environment
env = gym.make('CartPole-v1')

# Initialize REINFORCE agent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
learning_rate = 0.01
gamma = 0.99

agent = REINFORCE(state_dim, action_dim, learning_rate, gamma)

# Train the agent
agent.train(env, num_episodes=2000)
