import gym
from models import Actor, Critic
from sac import SAC

env = gym.make("Pendulum-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bounds = [env.action_space.low, env.action_space.high]
max_action = float(env.action_space.high[0])

actor = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim, action_dim)

sac_agent = SAC(
    state_dim=state_dim,
    action_dim=action_dim,
    action_bounds=action_bounds,
    actor_model=Actor,
    critic_model=Critic,
    actor_lr=1e-3,
    critic_lr=1e-3,
    alpha_lr=1e-3,
    gamma=0.99,
    tau=0.005,
    target_entropy=-action_dim,
    alpha=0.2,
    max_action=max_action
)

sac_agent.train(env, num_episodes=500, max_steps=200)
