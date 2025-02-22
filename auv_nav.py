import numpy as np
import random

# Define environment size
GRID_SIZE = 10
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Q-Table for storing state-action values
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 1.0
exploration_decay = 0.995
min_exploration_rate = 0.01

# Define reward function
def get_reward(x, y):
    if (x, y) == (GRID_SIZE - 1, GRID_SIZE - 1):
        return 100  # Goal reached
    return -1  # Every move has a small penalty

# Choose an action using an epsilon-greedy strategy
def choose_action(state):
    if random.uniform(0, 1) < exploration_rate:
        return random.choice(range(len(ACTIONS)))  # Explore
    return np.argmax(q_table[state[0], state[1]])  # Exploit

# Update Q-table
def update_q_table(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
    q_table[state[0], state[1], action] += learning_rate * (
        reward + discount_factor * q_table[next_state[0], next_state[1], best_next_action] - q_table[state[0], state[1], action]
    )

# Train the AI
def train_auv(episodes=1000):
    global exploration_rate
    for episode in range(episodes):
        x, y = 0, 0  # Start position
        while (x, y) != (GRID_SIZE - 1, GRID_SIZE - 1):
            action = choose_action((x, y))
            new_x, new_y = x, y

            if ACTIONS[action] == 'UP' and x > 0:
                new_x -= 1
            elif ACTIONS[action] == 'DOWN' and x < GRID_SIZE - 1:
                new_x += 1
            elif ACTIONS[action] == 'LEFT' and y > 0:
                new_y -= 1
            elif ACTIONS[action] == 'RIGHT' and y < GRID_SIZE - 1:
                new_y += 1

            reward = get_reward(new_x, new_y)
            update_q_table((x, y), action, reward, (new_x, new_y))
            x, y = new_x, new_y

        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)
        if episode % 100 == 0:
            print(f"Episode {episode}: Exploration Rate = {exploration_rate}")

    print("Training complete. Q-table updated.")

if __name__ == "__main__":
    train_auv()
    print("Q-Table:")
    print(q_table)
