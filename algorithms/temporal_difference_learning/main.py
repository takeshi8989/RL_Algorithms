import time
import numpy as np

from tic_tac_toe import TicTacToe
from sarsa import Sarsa
from expected_sarsa import ExpectedSarsa
from q_learning import QLearning
from double_q import DoubleQLearning
from n_step_td import NStepSarsa
from off_policy_nstep_td import OffPolicyNStepSarsa
from tree_backup import TreeBackup

# Initialize environment
env = TicTacToe()
n_states = 3**9  # 3^9 possible board configurations
n_actions = 9  # 9 possible board positions

# Initialize agents
sarsa_agent = Sarsa(n_states, n_actions, alpha=0.4, gamma=0.99, epsilon=0.4)
expected_sarsa_agent = ExpectedSarsa(n_states, n_actions)
q_learning_agent = QLearning(n_states, n_actions)
double_q_learning_agent = DoubleQLearning(n_states, n_actions)
nstep_sarsa_agent = NStepSarsa(n_states, n_actions)
off_policy_nstep_sarsa_agent = OffPolicyNStepSarsa(n_states, n_actions)
tree_backup_agent = TreeBackup(n_states, n_actions)

num_episodes = 3000000


def play_game_with_agent(env, agent):
    state = env.reset()
    done = False

    while not done:
        print("Agent is thinking...")
        time.sleep(3)

        action = agent.get_best_action(state)
        next_state, reward, done, info = env.step(action, user_play=True)

        env.render()

        if done:
            break

        user_action = int(input("Enter your action: "))
        state, reward, done, info = env.step(user_action, user_play=True)

        env.render()
        print("\n\n")

    print("Game over.")

    return reward


def evaluate_agent(agent, num_games):
    total_reward = 0

    env.reset()
    for _ in range(num_games):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(env, state)
            if action is None:
                break
            _, reward, done, _ = env.step(action)

            total_reward += reward

            if done:
                break

            opponent_action = np.random.choice(env.get_valid_actions())
            state, reward, done, _ = env.step(opponent_action)

            total_reward -= reward

    return total_reward / num_games


def train_agents(env, agent1, agent2, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        current_agent = agent1

        action = current_agent.choose_action(env, state)

        done = False
        while not done:
            next_state, reward, done, _ = env.step(action, user_play=True)
            if episode % 100000 == 0:
                env.render()
            next_action = current_agent.choose_action(env, next_state)

            current_agent.update_q(
                env, state, action, reward, next_state, next_action
            )

            if done:
                break

            state = next_state
            current_agent = agent1 if current_agent == agent2 else agent2
            action = current_agent.choose_action(env, state)


sarsa2_agent = Sarsa(n_states, n_actions, alpha=0.4, gamma=0.99, epsilon=0.4)
train_agents(env, sarsa_agent, sarsa2_agent, num_episodes)
while True:
    play_game_with_agent(env, sarsa_agent)
