import pygame
from collections import deque
import random
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box, Tuple

width = 10
height = 10
last_actions_size = 5

square_size = 40
screenwidth = width * square_size
screenheight = height * square_size


class Game(gymnasium.Env):

    def __init__(self, render_mode=None, FPS=10) -> None:
        super(Game, self).__init__()
        self.render_mode = render_mode
        self.steps = 0

        additional_info_size = 5

        ####
        ##  0 = up, 1 = right, 2 = down, 3 = left
        ####
        self.action_space = Discrete(4)

        # Observation space is the extra information + the snakes body
        self.observation_space = Box(
            low=-1,
            high=width,
            shape=(additional_info_size + width * height * 2,),
            dtype=np.float32,
        )

        self.reward_range = (min(-100, -(width + height)), 100)
        ## 0 = up, 1 = right, 2 = down, 3 = left
        self.direction = random.randint(0, 3)
        self.board = np.zeros((width, height))

        self.last_board = deque(maxlen=4)
        for i in range(4):
            self.last_board.append(self.board.copy())

        self.last_snake = deque(maxlen=4)
        ## Create a double linked list for the snake
        self.snake = deque(maxlen=width * height)
        self.eaten = False
        self.apple = (0, 0)
        self.reseted = False
        self.reward_var = 0

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

        # Store the last 4 boards
        self.last_board.append(self.board.copy())
        self.last_snake.append(self.snake[0])

        if self.steps > 100:
            self.truncated = True
            self.reward_var = -100
            self.steps = 0

        if self.reward_var is None:
            # snake_head = self.snake[0]
            # apple = self.apple
            # self.reward_var = -(
            #    abs(apple[0] - snake_head[0]) + abs(apple[1] - snake_head[1])
            # )
            self.reward_var = -self.steps / 20

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
        head = self.snake[0]
        # Check collsion with border
        if head[0] < 0 or head[0] >= width or head[1] < 0 or head[1] >= height:
            # print("Collision with border : reset")
            self.reward_var = -100
            # self.reset()
            self.reseted = True
        elif self.board[head[0]][head[1]] == 2:
            # print("apple eaten")
            self.reward_var = 100
            self.steps = 0
            self.eatApple(poped=poped)

        elif self.board[head[0]][head[1]] == 1:
            # print("Collision with snake : reset")
            self.reward_var = -100
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

        action = int(action)
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
        x = self.snake[0][0]
        y = self.snake[0][1]

        if self.direction == 1:
            self.snake.appendleft((x + 1, y))
            poped = self.snake.pop()
        elif self.direction == 3:
            self.snake.appendleft((x - 1, y))
            poped = self.snake.pop()
        elif self.direction == 0:
            self.snake.appendleft((x, y - 1))
            poped = self.snake.pop()
        elif self.direction == 2:
            self.snake.appendleft((x, y + 1))
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

        self.snake.clear()
        self.snake.append((random.randint(1, width - 2), random.randint(1, height - 2)))
        self.direction = random.randint(0, 3)

        self.steps = 0
        self.board = np.zeros((width, height))

        self.board[self.snake[0][0]][self.snake[0][1]] = 1
        self.spawnApple()

        self.last_board.clear()
        for i in range(4):
            self.last_board.append(self.board.copy())
        self.last_snake.clear()

        state = self.getState()

        self.reseted = False
        self.truncated = False
        self.render()
        return state, {}

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def getState(self):
        # extra info with the snake position and the apple position and the direction
        additional_info = np.array(
            [
                self.snake[0][0],
                self.snake[0][1],
                self.apple[0],
                self.apple[1],
                self.direction,
            ],
            dtype=np.float32,
        )

        # Convert the deque to a numpy array and get rid of tuples
        fixed_length_snake_array = np.zeros((width * height * 2,), dtype=np.float32)
        # Make default value -1
        fixed_length_snake_array.fill(-1)

        for i, s in enumerate(self.snake):
            fixed_length_snake_array[i] = s[0]
            fixed_length_snake_array[i + 1] = s[1]

        concat = np.concatenate([additional_info, fixed_length_snake_array])

        return concat


if __name__ == "__main__":
    env = Game(render_mode="human", FPS=1)
    env.reset()
    while True:
        action = env.setDirectionWithKeys()
        obs, reward, reseted, truncated, info = env.step(action)
        if reseted or truncated:
            env.reset()
        env.render()
    env.close()
