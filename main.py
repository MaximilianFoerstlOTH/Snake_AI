from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from game import Game
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import BaseCallback
import torch

# Create the environment
#env = Game(render_mode="rgb_array", FPS=10)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional values in TensorBoard.
    """

    def __init__(self, log_dir, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = None

    def _on_training_start(self) -> None:
        if self.writer is None:
            self.writer = torch.utils.tensorboard.SummaryWriter(self.log_dir)

    def _on_step(self) -> bool:
        # Log the reward at each step
        reward = self.locals['rewards'][0]
        self.writer.add_scalar('reward', reward, self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


register(
    id='SnakeGame-v0',
    entry_point='game:Game',
    max_episode_steps=1000,
)


env = gymnasium.make('SnakeGame-v0')

# It will check your custom environment and output additional warnings if needed
check_env(env)

#env = DummyVecEnv([lambda: env])

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=100):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, observations):
        return self.cnn(observations)

# Create the model

### CNN model
#model = DQN("CnnPolicy", env, verbose=1, tensorboard_log="./dqn_snake_tensorboard/", policy_kwargs={"features_extractor_class": CustomCNN, "features_extractor_kwargs": {"features_dim": 128}})
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_snake_tensorboard_discrete/")

callback = TensorboardCallback("./dqn_snake_tensorboard_discrete/")

#Train the agent
#model.learn(total_timesteps=100000, progress_bar=True, callback=callback)
#model.save("dqn_snake")
#del model

env = Game(render_mode="human", FPS=20)
#env = gymnasium.make('SnakeGame-v0')

model = DQN.load("dqn_snake", env=env)
env = model.get_env()

# Test the trained agent
obs = env.reset()


for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, reseted, _ = env.step(action)
    env.render()
    if reseted :
        obs = env.reset()