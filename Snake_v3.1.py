import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize Pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 600, 600
CELL_SIZE = 20
FPS = 10  # Slowed down to visualize movements

# Colors (Updated to Match Reference Style)
BACKGROUND_GREEN = (170, 204, 102)
SNAKE_COLOR = (0, 0, 0)
FOOD_COLOR = (0, 0, 0)
BORDER_COLOR = (0, 0, 0)

# Directions
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, Right, Up, Down

# Pygame Setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

# Neural Network Model
class SnakeNet(nn.Module):
    def __init__(self):
        super(SnakeNet, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)  # 4 outputs for 4 directions
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation, raw scores
        return x

# Fitness Function
class SnakeAI:
    def __init__(self):
        self.snake = [(WIDTH // 2, HEIGHT // 2)]
        self.food = self.spawn_food()
        self.direction = random.choice(DIRECTIONS)
        self.score = 0
        self.alive = True
        self.brain = SnakeNet()
    
    def spawn_food(self):
        return (random.randrange(0, WIDTH, CELL_SIZE), random.randrange(0, HEIGHT, CELL_SIZE))
    
    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        obstacles = [
            1 if (head_x - CELL_SIZE, head_y) in self.snake or head_x - CELL_SIZE < 0 else 0,
            1 if (head_x + CELL_SIZE, head_y) in self.snake or head_x + CELL_SIZE >= WIDTH else 0,
            1 if (head_x, head_y - CELL_SIZE) in self.snake or head_y - CELL_SIZE < 0 else 0,
            1 if (head_x, head_y + CELL_SIZE) in self.snake or head_y + CELL_SIZE >= HEIGHT else 0,
        ]
        direction_encoding = [int(self.direction == d) for d in DIRECTIONS]
        return torch.tensor([direction_encoding + [food_x - head_x, food_y - head_y] + obstacles], dtype=torch.float32)
    
    def choose_direction(self):
        state = self.get_state()
        with torch.no_grad():
            output = self.brain(state)
        proposed_direction = DIRECTIONS[torch.argmax(output).item()]
        
        # Prevent immediate reversals
        if proposed_direction == (-self.direction[0], -self.direction[1]):
            return self.direction  # Keep moving in the same direction
        return proposed_direction
    
    def move(self):
        if not self.alive:
            return
        self.direction = self.choose_direction()
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0] * CELL_SIZE, head_y + self.direction[1] * CELL_SIZE)
        
        if new_head in self.snake or not (0 <= new_head[0] < WIDTH and 0 <= new_head[1] < HEIGHT):
            self.alive = False
            return
        
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 10
            self.food = self.spawn_food()
        else:
            self.snake.pop()
        
        self.score += 0.5  # Reward for survival

# Run Simulation with Visualization
population_size = 10  # Reduced to improve visualization
snakes = [SnakeAI() for _ in range(population_size)]

def draw_game():
    screen.fill(BACKGROUND_GREEN)
    pygame.draw.rect(screen, BORDER_COLOR, (0, 0, WIDTH, HEIGHT), 5)  # Border
    
    for snake in snakes:
        if snake.alive:
            for segment in snake.snake:
                pygame.draw.rect(screen, SNAKE_COLOR, (segment[0], segment[1], CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, FOOD_COLOR, (snake.food[0], snake.food[1], CELL_SIZE, CELL_SIZE))
    
    pygame.display.flip()
    clock.tick(FPS)

def run_generation():
    global snakes
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        alive_snakes = [s for s in snakes if s.alive]
        if not alive_snakes:
            break  # If all snakes are dead, stop the generation
        
        for snake in alive_snakes:
            snake.move()
        
        draw_game()
    
    # Selection: Choose top-performing snakes
    snakes.sort(key=lambda s: s.score, reverse=True)
    
    # Crossover & Mutation - Evolving Neural Networks
    top_performers = snakes[:5]
    new_population = []
    for _ in range(population_size - 1):  # Keep one elite
        parent = random.choice(top_performers)
        offspring = SnakeAI()
        offspring.brain.load_state_dict(parent.brain.state_dict())
        with torch.no_grad():
            for param in offspring.brain.parameters():
                param += torch.randn_like(param) * 0.1  # Mutate weights
        new_population.append(offspring)
    
    new_population.append(top_performers[0])  # Keep best performer
    snakes = new_population

# Evolve over generations with visualization
for generation in range(10):
    print(f"Generation {generation} - Best Score: {max(snake.score for snake in snakes)}")
    run_generation()