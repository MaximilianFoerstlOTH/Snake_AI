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
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.ppo.policies import CnnPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy

# Create the environment
# env = Game(render_mode="rgb_array", FPS=10)


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
        reward = self.locals["rewards"][0]
        self.writer.add_scalar("reward", reward, self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


register(
    id="SnakeGame-v0",
    entry_point="game:Game",
    max_episode_steps=1000,
)
# env = gymnasium.make("SnakeGame-v0")
env = Game(render_mode="rgb_array", FPS=10)

# It will check your custom environment and output additional warnings if needed
check_env(env)


def make_env():
    return Game(render_mode="rgb_array", FPS=10)


env = make_vec_env(make_env, n_envs=1)


class CustomCNN(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gymnasium.spaces.Box, features_dim: int = 512
    ):

        # The shape of the observation space will be (channels, height, width)
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # Define your custom CNN architecture here
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6272, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.extra_info = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )

        # Define the fully connected layer that will produce the final features
        self.linear = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        image = observations[:, : (144 * 4)]
        info = observations[:, (4 * 144) :]

        # Reshape the image
        image = image.reshape(-1, 4, 12, 12)

        # Pass through the CNN layers
        x = self.cnn(image)

        # Pass through the extra info layers
        x_extra_info = self.extra_info(info)

        # Concatenate the output of the CNN with the extra info
        x = torch.cat([x, x_extra_info], dim=1)

        # Pass through the fully connected layers
        return self.linear(x)


class CustomCnnPolicy2(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomCnnPolicy2, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=64),
        )


model = PPO(
    CustomCnnPolicy2,
    env,
    verbose=1,
    tensorboard_log="./dqn_snake_tensorboard_discrete/",
)

callback = TensorboardCallback("./dqn_snake_tensorboard_discrete/")


#### Train the agent

model.learn(total_timesteps=9_00_000, progress_bar=True, callback=callback)
model.save("dqn_snake")
# del model

env = Game(render_mode="human", FPS=20)
# env = gymnasium.make('SnakeGame-v0')

model = PPO.load("dqn_snake", env=env)
env = model.get_env()

# Test the trained agent
obs = env.reset()


for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, reseted, _ = env.step(action)
    env.render()
    if reseted:
        obs = env.reset()
