import gym
from ddpg import DDPG

env = gym.make("Pendulum-v1")  # Replace with your desired environment

agent = DDPG(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    action_low=env.action_space.low,
    action_high=env.action_space.high,
)

agent.train(env, num_episodes=2000, max_steps=200)
