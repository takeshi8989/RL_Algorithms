import gym
from dqn import DQNAgent
import numpy as np

env = gym.make('CartPole-v1')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)
num_episodes = 5000

for episode in range(num_episodes):
    # Extract the state from the tuple returned by env.reset()
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)  # Ensure state is a NumPy array
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        # Ensure next_state is a NumPy array
        next_state = np.array(next_state, dtype=np.float32)
        done = terminated or truncated

        # Store the transition
        agent.store_transition(
            state.tolist(), action, reward, next_state.tolist(), done
        )

        agent.train()

        state = next_state
        total_reward += reward

    print(f'Episode: {episode + 1}, Total Reward: {total_reward}')

    # Update the target network periodically
    if episode % 10 == 0:
        agent.update_target_network()

    # Decay epsilon after each episode
    agent.decay_epsilon()
