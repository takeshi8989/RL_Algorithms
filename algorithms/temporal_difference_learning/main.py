"""
This is not working as good as I want it to. 
The agent is not learning to play the game well.
And we cannot use Sarsa, NStepSarsa, OffPolicyNStepSarsa, TreeBackup yet.
"""

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
sarsa_agent = Sarsa(n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1)
expected_sarsa_agent = ExpectedSarsa(n_states, n_actions)
q_learning_agent = QLearning(n_states, n_actions)
double_q_learning_agent = DoubleQLearning(n_states, n_actions)
nstep_sarsa_agent = NStepSarsa(n_states, n_actions)
off_policy_nstep_sarsa_agent = OffPolicyNStepSarsa(n_states, n_actions)
tree_backup_agent = TreeBackup(n_states, n_actions)


def evaluate_agent(agent, num_games):
    win = 0
    draw = 0
    lose = 0

    env.reset()
    for _ in range(num_games):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_best_action(env, state)
            if action is None:
                break
            _, reward, done, _ = env.step(action)

            if done:
                break

            opponent_action = np.random.choice(env.get_valid_actions())
            state, reward, done, _ = env.step(opponent_action)

            if done:
                reward = -1

        if reward == 1:
            win += 1
        elif reward == 0:
            draw += 1
        else:
            lose += 1

    return win, draw, lose


def train_agent(env, agent, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()  # Reset the environment for a new game
        done = False
        agent_turn = True  # Track whose turn it is

        while not done:
            if agent_turn:
                # Agent's move
                action = agent.choose_action(env, state)
                next_state, reward, done, _ = env.step(action)

                if not done:
                    agent.update_q(env, state, action, -0.01, next_state)
                else:
                    agent.update_q(env, state, action, reward, next_state)

                state = next_state
                agent_turn = False

            else:
                # Opponent's move (random valid action)
                opponent_action = np.random.choice(env.get_valid_actions())
                next_state, reward, done, _ = env.step(opponent_action)

                if done:
                    agent.update_q(env, state, action, -1, next_state)

                state = next_state
                agent_turn = True


agent = q_learning_agent

num_episodes = 100000
agent.epsilon = 1.0  # Start with a high exploration rate
epsilon_decay = 0.99995  # Decay epsilon slowly

for episode in range(num_episodes):
    train_agent(env, agent, 1)
    agent.epsilon = max(0.1, agent.epsilon * epsilon_decay)

    if episode % 10000 == 0:
        print(f"Episode {episode}, epsilon: {agent.epsilon:.4f}")

# Evaluate the agent's performance after training
print("Win, Draw, Lose:", evaluate_agent(agent, 1000))
