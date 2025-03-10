import pygame
import random
import numpy as np
import time

# Initialize Pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 600, 600
TOP_BAR_HEIGHT = 50
SIDE_BAR_WIDTH = 200
CELL_SIZE = 20
FPS = 60

# Colors
WHITE = (255, 255, 255)
BACKGROUND_GRAY = (70, 70, 70)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)  # Highlight best snake

# Directions
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Screen Setup
screen = pygame.display.set_mode((WIDTH, HEIGHT + TOP_BAR_HEIGHT))
pygame.display.set_caption("Snake GA Playground")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

# Global Time Tracker
generation_start_time = time.time()

# AI-Controlled Snake Class
class SnakeAI:
    def __init__(self, brain=None):
        self.snake = [(WIDTH // 2, HEIGHT // 2)]
        self.food = self.spawn_food()
        self.direction = random.choice(DIRECTIONS)
        self.score = 0
        self.length = 0
        self.alive = True
        self.start_time = time.time()
        self.previous_positions = []

        self.brain = brain if brain is not None else np.random.rand(7) * 2 - 1

    def spawn_food(self):
        while True:
            food_x = random.randrange(0, WIDTH, CELL_SIZE)
            food_y = random.randrange(0, HEIGHT, CELL_SIZE)
            if (food_x, food_y) not in self.snake:
                return food_x, food_y

    def choose_direction(self):
        head_x, head_y = self.snake[0]
        move_scores = []
        valid_moves = []

        for move in DIRECTIONS:
            new_x, new_y = head_x + move[0] * CELL_SIZE, head_y + move[1] * CELL_SIZE
            if (new_x, new_y) in self.snake or not (0 <= new_x < WIDTH and 0 <= new_y < HEIGHT):
                continue  # Skip invalid moves

            open_space = sum(1 for m in DIRECTIONS if (new_x + m[0] * CELL_SIZE, new_y + m[1] * CELL_SIZE) not in self.snake)
            move_score = (
                self.brain[0] * (100 if (new_x, new_y) == self.food else 0)
                + self.brain[1] * -abs(new_x - self.food[0]) + abs(new_y - self.food[1])
                - self.brain[2] * (1 if (new_x, new_y) in self.previous_positions else 0)
                - self.brain[3] * (1 if (new_x, new_y) == head_x - move[0] * CELL_SIZE else 0)
                + self.brain[4] * open_space
            )
            move_scores.append(move_score)
            valid_moves.append(move)

        return valid_moves[np.argmax(move_scores)] if valid_moves else random.choice(DIRECTIONS)

    def move(self):
        if not self.alive:
            return
        self.direction = self.choose_direction()
        new_head = (self.snake[0][0] + self.direction[0] * CELL_SIZE, self.snake[0][1] + self.direction[1] * CELL_SIZE)

        if new_head in self.snake or not (0 <= new_head[0] < WIDTH and 0 <= new_head[1] < HEIGHT):
            self.alive = False
            return

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.length += 1
            self.food = self.spawn_food()
        else:
            self.snake.pop()

def draw_game():
    screen.fill(BACKGROUND_GRAY)
    for snake in snakes:
        if snake.alive:
            for segment in snake.snake:
                pygame.draw.rect(screen, GREEN, (*segment, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, RED, (*snake.food, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(FPS)

def run_generation():
    while any(s.alive for s in snakes):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        for snake in snakes:
            snake.move()
        draw_game()

population_size = 50
snakes = [SnakeAI() for _ in range(population_size)]
for generation in range(10):
    print(f"Generation {generation}")
    run_generation()
