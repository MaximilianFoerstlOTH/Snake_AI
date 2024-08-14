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
# env = Game(render_mode="rgb_array", FPS=10)

# It will check your custom environment and output additional warnings if needed
# check_env(env)


def make_env():
    return Game(render_mode="rgb_array", FPS=10)


env = make_vec_env(make_env, n_envs=1)


class RNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gymnasium.spaces.Box, features_dim: int = 512
    ):

        # The shape of the observation space will be (channels, height, width)
        super(RNNFeatureExtractor, self).__init__(observation_space, features_dim)

        # Process the body of the snake
        self.rnn = nn.LSTM(
            input_size=10 * 10 * 2, hidden_size=32, num_layers=1, batch_first=True
        )

        self.extra_info = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )

        # Define the fully connected layer that will produce the final features
        self.linear = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, features_dim), nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract the extra information from 0 to 5
        info = observations[..., :5]
        # Extract the body of the snake
        snake_body = observations[..., 5:]

        # Process the body of the snake
        x = self.rnn(snake_body)

        # Pass through the extra info layers
        x_extra_info = self.extra_info(info)

        # Concatenate the output of the RNN with the extra info
        x = torch.cat([x[0], x_extra_info], dim=1)

        # Pass through the fully connected layers
        return self.linear(x)


class CustomRNNPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomRNNPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=RNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64),
        )


model = PPO(
    CustomRNNPolicy,
    env,
    verbose=1,
    tensorboard_log="./dqn_snake_tensorboard_discrete/",
)

callback = TensorboardCallback("./dqn_snake_tensorboard_discrete/")


#### Train the agent

# model.learn(total_timesteps=6_000_000, progress_bar=True, callback=callback)
# model.save("dqn_snake")
# del model

env = Game(render_mode="human", FPS=20)
# env = gymnasium.make('SnakeGame-v0')

model = PPO.load("dqn_snake", env=env)
env = model.get_env()

# Test the trained agent
obs = env.reset()
lstm_states = None

for i in range(10000):
    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
    obs, rewards, reseted, _ = env.step(action)
    env.render()
    if reseted:
        obs = env.reset()
