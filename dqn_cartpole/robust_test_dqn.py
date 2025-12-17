import gymnasium as gym
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# =========================
# CONFIG
# =========================
ENV_NAME = "CartPole-v1"
MODEL_PATH = "dqn_cartpole_SOLVED.h5"   # WAJIB model solved

MAX_STEPS = 500
DISTURB_STEP = 200
DISTURB_ANGLE = 0.15        # rad
RECOVERY_THRESHOLD = 0.05   # rad (dianggap stabil kembali)
SLEEP_TIME = 0.02

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
# INIT ENV & MODEL
# =========================
env = gym.make(ENV_NAME, render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = build_model(state_size, action_size)
model.load_weights(MODEL_PATH)

state, _ = env.reset()
state = state.reshape(1, -1)

angles = []
steps = []

disturbed = False
recovered = False
recovery_step = None

print("[INFO] Robustness Test Started")
print(f"[INFO] Disturbance injected at step {DISTURB_STEP}")

# =========================
# MAIN LOOP
# =========================
for t in range(MAX_STEPS):

    q_values = model(state, training=False).numpy()
    action = np.argmax(q_values[0])

    next_state, reward, terminated, truncated, _ = env.step(action)

    # ===== Inject disturbance =====
    if t == DISTURB_STEP:
        print("[DISTURBANCE] Pole angle disturbed!")
        next_state[2] += DISTURB_ANGLE
        disturbed = True

    angle = next_state[2]
    angles.append(angle)
    steps.append(t)

    # ===== Check recovery =====
    if disturbed and not recovered:
        if abs(angle) < RECOVERY_THRESHOLD:
            recovered = True
            recovery_step = t
            print(f"[RECOVERY] System recovered at step {t}")

    state = next_state.reshape(1, -1)

    env.render()
    time.sleep(SLEEP_TIME)

    if terminated:
        print(f"[FAILURE] Pole fell at step {t}")
        break

    if truncated:
        print(f"[TIME LIMIT] Episode ended at step {t}")
        break

env.close()

# =========================
# PLOT RESPONSE
# =========================
plt.figure(figsize=(10, 4))
plt.plot(steps, angles, label="Pole Angle (rad)")
plt.axvline(DISTURB_STEP, color="r", linestyle="--", label="Disturbance")

if recovered:
    plt.axvline(recovery_step, color="g", linestyle="--", label="Recovery")

plt.axhline(RECOVERY_THRESHOLD, color="k", linestyle=":", alpha=0.4)
plt.axhline(-RECOVERY_THRESHOLD, color="k", linestyle=":", alpha=0.4)

plt.xlabel("Time Step")
plt.ylabel("Pole Angle (rad)")
plt.title("DQN Robustness Test: Disturbance & Recovery")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.show(block=False)
plt.pause(5)   # tampil 5 detik
plt.close()


# =========================
# SUMMARY
# =========================
print("\n========== ROBUSTNESS SUMMARY ==========")
print(f"Disturbance step : {DISTURB_STEP}")
print(f"Disturbance mag  : {DISTURB_ANGLE} rad")

if recovered:
    print(f"Recovery step    : {recovery_step}")
    print(f"Recovery time    : {recovery_step - DISTURB_STEP} steps")
    print("Robustness       : ✅ PASS")
else:
    print("Recovery         : ❌ FAILED")
    print("Robustness       : ❌ FAIL")

print("=======================================\n")
