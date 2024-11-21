import gym
from dqn import DQNAgent
import numpy as np

env = gym.make('CartPole-v1')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)
num_episodes = 5000
update_target_every = 10

for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        done = terminated or truncated

        # Store the transition
        agent.store_transition(
            state, action, reward, next_state, done
        )

        agent.train()

        state = next_state
        total_reward += reward

    print(f'Episode: {episode + 1}, Total Reward: {total_reward}')

    # Update the target network periodically
    if episode % update_target_every == 0:
        agent.update_target_network()

    # Decay epsilon after each episode
    agent.decay_epsilon()
