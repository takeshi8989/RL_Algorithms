import gym
from ppo import PPO

# Initialize environment and PPO agent
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPO(
    state_dim=state_dim,
    action_dim=action_dim,
    clip_epsilon=0.2,
    gamma=0.99,
    actor_lr=1e-4,
    critic_lr=1e-3,
    epochs=10,
    batch_size=64
)

# Train the agent
agent.train(env, num_iterations=100, timesteps_per_batch=2048)
