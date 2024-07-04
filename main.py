from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from game import Game
import numpy as np

# Create the environment
env = Game(render_game=False, FPS=10)

check_env(env)


# Create the model
model = DQN("MlpPolicy", env, verbose=1)

# Train the agent
# model.learn(total_timesteps=10000)
# model.save("dqn_snake")

model.load("dqn_snake")
# Test the trained agent
obs = env.reset()
env.FPS = 10
env.render_game = True
while True:
    action, _states = model.predict(np.array(obs), deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
