import pygame
from collections import deque
import random
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box

width = 10
height = 10
last_actions_size = 5

square_size = 40
screenwidth = width * square_size
screenheight = height * square_size


class Game(gymnasium.Env):

    def __init__(self, render_mode=None, FPS=10) -> None:
        self.render_mode = render_mode
        self.steps = 0
        self.action_space = Discrete(4, start=0)
        ## Observation space is the board
        # self.observation_space = Box(low = np.array([0,0,0,0,0, 0, 0,0,0,0,0], dtype=np.float32), high = np.array([width, height,width, height,3, width * height, width + height,3,3,3,3], dtype=np.float32), dtype=np.float32)
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array(
                [width, height, 3, width, height, 1, 1, 1, 1],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.reward_range = (min(-1000, -(width + height)), 10)
        ## 0 = up, 1 = right, 2 = down, 3 = left
        self.direction = random.randint(0, 3)
        self.board = np.zeros((width, height))

        ## Create a double linked list for the snake
        self.snake = deque()
        self.eaten = False
        self.apple = (0, 0)
        self.reseted = False
        self.reward_var = 0
        self.last_actions = deque([0, 0, 0, 0, 0], maxlen=last_actions_size)

        ##Setup pygame
        self.running = True
        self.FPS = FPS
        self.truncated = False

        # self.screen = pygame.display.set_mode((screenwidth, screenheight))
        self.screen = None
        self.clock = None
        # self.clock = pygame.time.Clock()

        self.reset()

    def step(self, action):
        ## make game mechanics
        self.reward_var = None
        self.move(action)
        self.steps += 1
        ## render game
        if self.render_mode == "human":
            self.render()

        # last two steps
        self.last_actions.append(action)

        if self.steps > 100:
            self.truncated = True
            self.reward_var = -1000
            self.steps = 0

        if self.reward_var is None:
            snake_head = self.snake[0]
            apple = self.apple
            self.reward_var = -(
                abs(apple[0] - snake_head[0]) + abs(apple[1] - snake_head[1])
            )

        return self.getState(), self.reward_var, self.reseted, self.truncated, {}

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screenwidth, screenheight))
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((screenwidth, screenheight))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        for x in range(width):
            for y in range(height):
                pygame.draw.rect(
                    self.screen,
                    # (255, 255, 255),
                    (0, 0, 0),
                    (x * square_size, y * square_size, square_size, square_size),
                    1,
                    border_radius=0,
                )
                if self.board[x][y] == 1:
                    pygame.draw.rect(
                        self.screen,
                        (0, 0, 0),
                        (x * square_size, y * square_size, square_size, square_size),
                    )
                if self.board[x][y] == 2:
                    pygame.draw.rect(
                        self.screen,
                        (255, 0, 0),
                        (x * square_size, y * square_size, square_size, square_size),
                    )

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.FPS)
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def spawnApple(self):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        while self.board[x][y] == 1 or self.board[x][y] == 2 or (x, y) in self.snake:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
        self.apple = (x, y)

        self.board[x][y] = 2

    def eatApple(self, poped):
        self.snake.append(poped)
        self.spawnApple()
        self.eaten = True

    def checkCollision(self, poped):
        # Check collsion with border
        if (
            self.snake[0][0] < 0
            or self.snake[0][0] >= width
            or self.snake[0][1] < 0
            or self.snake[0][1] >= height
        ):
            # print("Collision with border : reset")
            self.reward_var = -1000
            # self.reset()
            self.reseted = True
        elif self.board[self.snake[0][0]][self.snake[0][1]] == 2:
            # print("apple eaten")
            self.reward_var = 10
            self.steps = 0
            self.eatApple(poped=poped)

        elif self.board[self.snake[0][0]][self.snake[0][1]] == 1:
            # print("Collision with snake : reset")
            self.reward_var = -1000
            # self.reset()
            self.reseted = True

    def setDirectionWithKeys(self):

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return 0
        elif keys[pygame.K_DOWN]:
            return 2
        elif keys[pygame.K_RIGHT]:
            return 1
        elif keys[pygame.K_LEFT]:
            return 3

    def setDirection(self, action):
        if action == None:
            return
        if self.direction == 0 and action == 2:
            self.direction = 0
        elif self.direction == 2 and action == 0:
            self.direction = 2
        elif self.direction == 1 and action == 3:
            self.direction = 1
        elif self.direction == 3 and action == 1:
            self.direction = 3
        else:
            self.direction = action

    def move(self, action):
        ## Move snake deque
        self.setDirection(action=action)

        poped = []

        if self.direction == 1:
            self.snake.appendleft((self.snake[0][0] + 1, self.snake[0][1]))
            poped = self.snake.pop()
        elif self.direction == 3:
            self.snake.appendleft((self.snake[0][0] - 1, self.snake[0][1]))
            poped = self.snake.pop()
        elif self.direction == 0:
            self.snake.appendleft((self.snake[0][0], self.snake[0][1] - 1))
            poped = self.snake.pop()
        elif self.direction == 2:
            self.snake.appendleft((self.snake[0][0], self.snake[0][1] + 1))
            poped = self.snake.pop()

        self.eaten = False
        self.reseted = False
        ## Check collsion with apple and border
        self.checkCollision(poped)

        if not self.reseted:
            ## Move snake on the board
            self.board[self.snake[0][0]][self.snake[0][1]] = 1
            if not self.eaten:
                self.board[poped[0]][poped[1]] = 0

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.reseted = False
        self.truncated = False
        self.steps = 0
        self.snake.clear()
        self.board = np.zeros((width, height))

        self.snake.append((random.randint(1, width - 2), random.randint(1, height - 2)))
        self.direction = random.randint(0, 3)

        self.board[self.snake[0][0]][self.snake[0][1]] = 1
        self.spawnApple()

        self.render()

        return self.getState(), {}

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def getState(self):
        # get what is to the sides of the snake
        board_with_bounds = []
        board_with_bounds.append([1.0] * (width + 2))
        for line in self.board:
            board_with_bounds.append([1.0] + line.tolist() + [1.0])
        board_with_bounds.append([1.0] * (width + 2))

        top = 0
        right = 0
        bottom = 0
        left = 0
        if not self.reseted and not self.truncated:
            top = int(
                board_with_bounds[self.snake[0][0] + 1][self.snake[0][1] + 1 - 1] == 1
            )
            right = int(
                board_with_bounds[self.snake[0][0] + 1 + 1][self.snake[0][1] + 1] == 1
            )
            bottom = int(
                board_with_bounds[self.snake[0][0] + 1][self.snake[0][1] + 1 + 1] == 1
            )
            left = int(
                board_with_bounds[self.snake[0][0] + 1 - 1][self.snake[0][1] + 1] == 1
            )
        else:
            top = 1
            right = 1
            bottom = 1
            left = 1
        apple = self.apple

        # distance to apple
        distance = abs(apple[0] - self.snake[0][0]) + abs(apple[1] - self.snake[0][1])

        snake_head = self.snake[0]
        return np.array(
            [
                snake_head[0],
                snake_head[1],
                self.direction,
                # distance,
                # len(self.snake),
                apple[0],
                apple[1],
                # abs(apple[0] - snake_head[0]) + abs(apple[1] - snake_head[1]),
                top,
                right,
                left,
                bottom,
                # self.last_actions[0],
                # self.last_actions[1],
                # self.last_actions[2],
                # self.last_actions[3],
                # self.last_actions[4],
            ],
            dtype=np.float32,
        )

    # def reward(self, value=None):
    # reward is the distance to the apple
    # return -np.sqrt((self.snake[0][0] - self.apple[0]) ** 2 + (self.snake[0][1] - self.apple[1]) ** 2)


from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from game import Game
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import BaseCallback
import torch

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


env = gymnasium.make("SnakeGame-v0")


# env = DummyVecEnv([lambda: env])

# Create the model

### CNN model
model = PPO(
    "MlpPolicy", env, verbose=1, tensorboard_log="./dqn_snake_tensorboard_discrete/"
)

callback = TensorboardCallback("./dqn_snake_tensorboard_discrete/")

# Train the agent
model.learn(total_timesteps=10_000_000, progress_bar=True, callback=callback)
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
