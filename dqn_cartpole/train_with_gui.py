import gymnasium as gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# =====================================================
# CONFIG
# =====================================================
ENV_NAME = "CartPole-v1"

MAX_EPISODES = 2000
MAX_STEPS = 500
GAMMA = 0.99

EPSILON_START = 1.0
EPSILON_MIN   = 0.02
EPSILON_DECAY = 0.995

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 100_000

TARGET_UPDATE_FREQ = 1000

# =====================================================
# MODEL
# =====================================================
def build_model(state_size, action_size):
    model = models.Sequential([
        layers.Dense(128, activation="relu", input_shape=(state_size,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(action_size, activation="linear")
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    return model

# =====================================================
# INIT (GUI ENABLED)
# =====================================================
env = gym.make(ENV_NAME, render_mode="human")

state_size  = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = build_model(state_size, action_size)
target_net = build_model(state_size, action_size)
target_net.set_weights(policy_net.get_weights())

memory = deque(maxlen=MEMORY_SIZE)

epsilon = EPSILON_START
global_step = 0

print("[INFO] DQN TRAINING WITH GUI STARTED")

# =====================================================
# TRAINING LOOP (REAL-TIME GUI)
# =====================================================
for episode in range(1, MAX_EPISODES + 1):

    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    steps = 0

    for _ in range(MAX_STEPS):
        global_step += 1

        # ε-greedy
        if np.random.rand() < epsilon:
            action = random.randrange(action_size)
        else:
            q = policy_net(state, training=False).numpy()
            action = np.argmax(q[0])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        steps += 1

        # ==========================
        # LEARNING
        # ==========================
        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)

            states      = np.vstack([b[0] for b in batch])
            actions     = np.array([b[1] for b in batch])
            rewards     = np.array([b[2] for b in batch])
            next_states = np.vstack([b[3] for b in batch])
            dones       = np.array([b[4] for b in batch])

            target_q = policy_net(states, training=False).numpy()
            next_q   = target_net(next_states, training=False).numpy()

            for i in range(BATCH_SIZE):
                if dones[i]:
                    target_q[i][actions[i]] = rewards[i]
                else:
                    target_q[i][actions[i]] = rewards[i] + GAMMA * np.max(next_q[i])

            policy_net.train_on_batch(states, target_q)

        if global_step % TARGET_UPDATE_FREQ == 0:
            target_net.set_weights(policy_net.get_weights())

        if done:
            break

    # EPSILON DECAY
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    print(f"[GUI] Episode {episode:04d} | Steps: {steps:3d} | Epsilon: {epsilon:.3f}")

env.close()
print("[INFO] TRAINING FINISHED")
