from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from game import Game
import numpy as np
from stable_baselines3.common.env_util import make_vec_env


# Create the environment
env = Game(render_game=True, FPS=500)

# Create the model
model = DQN("MlpPolicy", env, verbose=1)

#Train the agent
model.learn(total_timesteps=10000)
model.save("dqn_snake")


env = Game(render_game=True, FPS=10)

model = DQN.load("dqn_snake")

# Test the trained agent
obs = env.reset()[0]

while True:
    action, _states = model.predict(np.array(obs), deterministic=True)
    obs, rewards, dones, info, _ = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
