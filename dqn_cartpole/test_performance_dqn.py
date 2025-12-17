import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# =========================
# CONFIG
# =========================
ENV_NAME = "CartPole-v1"
MODEL_PATH = "dqn_cartpole_SOLVED.h5"

NUM_EVAL_EPISODES = 20
MAX_STEPS = 500
SOLVE_THRESHOLD = 475

# =========================
# BUILD MODEL (SAMA DENGAN TRAINING)
# =========================
def build_model(state_size, action_size):
    model = models.Sequential([
        layers.Dense(128, activation="relu", input_shape=(state_size,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(action_size, activation="linear")
    ])
    return model

# =========================
# LOAD ENV & MODEL
# =========================
env = gym.make(ENV_NAME)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = build_model(state_size, action_size)
model.load_weights(MODEL_PATH)

print("[INFO] Loaded model weights from:", MODEL_PATH)
print("[INFO] Evaluation started (epsilon = 0, greedy policy)\n")

# =========================
# EVALUATION
# =========================
episode_steps = []

for ep in range(1, NUM_EVAL_EPISODES + 1):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])

    total_steps = 0

    for _ in range(MAX_STEPS):
        q_values = model(state, training=False).numpy()
        action = np.argmax(q_values[0])  # PURE GREEDY

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = np.reshape(next_state, [1, state_size])
        total_steps += 1

        if done:
            break

    episode_steps.append(total_steps)
    print(f"Episode {ep:02d} | Steps: {total_steps}")

# =========================
# SUMMARY
# =========================
episode_steps = np.array(episode_steps)

mean_steps = episode_steps.mean()
std_steps = episode_steps.std()
success_rate = np.mean(episode_steps >= SOLVE_THRESHOLD) * 100

print("\n========== PERFORMANCE SUMMARY ==========")
print(f"Episodes tested : {NUM_EVAL_EPISODES}")
print(f"Mean steps      : {mean_steps:.2f}")
print(f"Std steps       : {std_steps:.2f}")
print(f"Max steps       : {episode_steps.max()}")
print(f"Min steps       : {episode_steps.min()}")
print(f"Solve threshold : {SOLVE_THRESHOLD}")
print(f"Success rate (%) : {success_rate:.1f}")
print("========================================")

env.close()
