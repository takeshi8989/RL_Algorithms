import gym
from models import Actor, Critic
from td3 import TD3

env = gym.make("Pendulum-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bounds = [env.action_space.low, env.action_space.high]
max_action = float(env.action_space.high[0])

actor = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim, action_dim)

td3_agent = TD3(
    state_dim=state_dim,
    action_dim=action_dim,
    action_bounds=action_bounds,
    actor_model=Actor,
    critic_model=Critic,
    actor_lr=1e-3,
    critic_lr=1e-3,
    gamma=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_delay=2,
    buffer_size=1000000,
    batch_size=256,
    max_action=max_action
)

td3_agent.train(env, num_episodes=100, max_steps=200, noise_scale=0.1)
