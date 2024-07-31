import pygame
from collections import deque
import random
import gymnasium
import numpy as np

width = 10
height = 10

square_size = 40
screenwidth = width * square_size
screenheight = height * square_size


class Game(gymnasium.Env):

    def __init__(self, render_mode=None, FPS=10) -> None:
        self.render_mode = render_mode
        self.steps = 0
        self.action_space = gymnasium.spaces.Discrete(3)
        ## Observation space is the board
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(width,height,), dtype=int
        )
        self.reward_range = (min(-10, -width), 100)
        ## 0 = up, 1 = right, 2 = down, 3 = left
        self.direction = random.randint(0, 3)
        self.board = np.zeros((width, height))

        ## Create a double linked list for the snake
        self.snake = deque()
        self.eaten = False
        self.apple = (0, 0)
        self.reseted = False
        self.reward_var = 0

        ##Setup pygame
        self.running = True
        self.FPS = FPS
        self.truncated = False

        #self.screen = pygame.display.set_mode((screenwidth, screenheight))
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

        if self.steps > 100:
            self.truncated = True
            self.steps = 0 

        if self.reward_var is None:
            self.reward_var = -min(abs(self.apple[0] - self.snake[0][0]), abs(self.apple[1] - self.snake[0][1]))

        return np.array(self.board, dtype=np.int64), self.reward_var, self.reseted, self.truncated, {}

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


        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.screen.fill((255, 255, 255))

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
            print("Collision with border : reset")
            self.reward_var = -10
            #self.reset()
            self.reseted = True
        elif self.board[self.snake[0][0]][self.snake[0][1]] == 2:
            print("apple eaten")
            self.reward_var = 100
            self.steps = 0
            self.eatApple(poped=poped)

        elif self.board[self.snake[0][0]][self.snake[0][1]] == 1:
            print("Collision with snake : reset")
            self.reward_var = -10
            #self.reset()
            self.reseted = True

    def setDirectionWithKeys(self):
        

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.direction = (self.direction + 3) % 4
        if keys[pygame.K_RIGHT]:
            self.direction = (self.direction + 1) % 4
        if keys[pygame.K_UP]:
            self.direction = (self.direction + 0) % 4

    def setDirection(self, turn):
        # 0 = turn right, 1 = stay, 2 = turn left

        if turn == 0:
            self.direction = (self.direction + 1) % 4
        elif turn == 2:
            self.direction = (self.direction + 3) % 4

    def move(self, action):
        # self.setDirectionWithKeys()
        ## Move snake deque
        self.setDirection(action)
        poped = []
        # Move with left, keep, right
        # NOT with left, up, right, down
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

        return np.array(self.board, dtype=np.int64), {}

        # Set direction to forward

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    #def reward(self, value=None):
        # reward is the distance to the apple
        #return -np.sqrt((self.snake[0][0] - self.apple[0]) ** 2 + (self.snake[0][1] - self.apple[1]) ** 2)
