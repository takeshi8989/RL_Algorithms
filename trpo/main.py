import gym
from trpo import TRPO

env = gym.make("LunarLander-v2")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

gamma = 0.995
delta = 0.001
alpha = 0.3
max_backtracking_steps = 10
timesteps_per_batch = 2048
num_iterations = 500

agent = TRPO(
    state_dim=state_dim,
    action_dim=action_dim,
    gamma=gamma,
    delta=delta,
    alpha=alpha,
    max_backtracking_steps=max_backtracking_steps
)

agent.train(
    env,
    num_iterations=num_iterations,
    timesteps_per_batch=timesteps_per_batch
)

env.close()
