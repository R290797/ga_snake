import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 600, 600
CELL_SIZE = 20
FPS = 60

# Colors
WHITE = (255, 255, 255)
BACKGROUND_GRAY = (70, 70, 70)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Directions
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, Right, Up, Down

# Pygame Setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

# Fitness Function
class SnakeAI:
    def __init__(self):
        self.snake = [(WIDTH // 2, HEIGHT // 2)]
        self.food = self.spawn_food()
        self.direction = random.choice(DIRECTIONS)
        self.score = 0
        self.alive = True
        self.brain = np.random.rand(4)  # AI decision weights
    
    def spawn_food(self):
        return (random.randrange(0, WIDTH, CELL_SIZE), random.randrange(0, HEIGHT, CELL_SIZE))
    
    def move(self):
        if not self.alive:
            return
        head_x, head_y = self.snake[0]
        
        # AI Decision: Choose best move
        self.direction = self.choose_direction()
        
        new_head = (head_x + self.direction[0] * CELL_SIZE, head_y + self.direction[1] * CELL_SIZE)
        
        # Collision Detection
        if new_head in self.snake or not (0 <= new_head[0] < WIDTH and 0 <= new_head[1] < HEIGHT):
            self.alive = False
            return
        
        self.snake.insert(0, new_head)
        
        # Check for food
        if new_head == self.food:
            self.score += 10
            self.food = self.spawn_food()
        else:
            self.snake.pop()
    
    def choose_direction(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        distances = [
            abs(food_x - (head_x - CELL_SIZE)) + abs(food_y - head_y),  # Left
            abs(food_x - (head_x + CELL_SIZE)) + abs(food_y - head_y),  # Right
            abs(food_x - head_x) + abs(food_y - (head_y - CELL_SIZE)),  # Up
            abs(food_x - head_x) + abs(food_y - (head_y + CELL_SIZE)),  # Down
        ]
        
        return DIRECTIONS[np.argmin(distances)]  # Move toward the closest option
    
# Run Simulation with Visualization
population_size = 50
snakes = [SnakeAI() for _ in range(population_size)]

def run_generation():
    global snakes
    running = True
    while running:
        screen.fill(BACKGROUND_GRAY)
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        alive_snakes = [s for s in snakes if s.alive]
        if not alive_snakes:
            break  # If all snakes are dead, stop the generation
        
        for snake in alive_snakes:
            snake.move()
            
            # Draw Snake
            for segment in snake.snake:
                pygame.draw.rect(screen, GREEN, (segment[0], segment[1], CELL_SIZE, CELL_SIZE))
            
            # Draw Food
            pygame.draw.rect(screen, RED, (snake.food[0], snake.food[1], CELL_SIZE, CELL_SIZE))
            
        pygame.display.flip()
    
    # Selection: Choose top-performing snakes
    snakes.sort(key=lambda s: s.score, reverse=True)
    
    # Crossover & Mutation
    top_performers = snakes[:10]
    new_population = []
    for _ in range(population_size - 1):  # Keep one elite
        parent = random.choice(top_performers)
        offspring = SnakeAI()
        offspring.brain = parent.brain + np.random.randn(4) * 0.1  # Mutate
        new_population.append(offspring)
    
    new_population.append(top_performers[0])  # Keep best performer
    snakes = new_population

# Evolve over generations with visualization
for generation in range(10):
    print(f"Generation {generation} - Best Score: {max(snake.score for snake in snakes)}")
    run_generation()