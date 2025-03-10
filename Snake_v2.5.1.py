import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 600, 600
TOP_BAR_HEIGHT = 50
SIDE_BAR_WIDTH = 200
PADDING = 20  # Variable padding for the game grid
CELL_SIZE = 20
GAP = 2
FPS = 60

# Colors
WHITE = (255, 255, 255)
BACKGROUND_GRAY = (70, 70, 70)
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Directions
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, Right, Up, Down

# Screen Configurations
screen = pygame.display.set_mode((WIDTH + SIDE_BAR_WIDTH, HEIGHT + TOP_BAR_HEIGHT))
pygame.display.set_caption("Snake GA Playground")
clock = pygame.time.Clock()

# Font setup
font = pygame.font.SysFont(None, 30)

# Adjust game area with padding
GAME_AREA_X = PADDING
GAME_AREA_Y = PADDING + TOP_BAR_HEIGHT
GAME_AREA_WIDTH = WIDTH - 2 * PADDING
GAME_AREA_HEIGHT = HEIGHT - 2 * PADDING

# Fitness Function
class SnakeAI:
    def __init__(self):
        self.snake = [(GAME_AREA_X + GAME_AREA_WIDTH // 2, GAME_AREA_Y + GAME_AREA_HEIGHT // 2)]
        self.food = self.spawn_food()
        self.direction = random.choice(DIRECTIONS)
        self.score = 0
        self.length = 0  # Tracks total food eaten
        self.alive = True
        self.brain = np.random.rand(4)  # AI decision weights
        self.previous_positions = []  # Store previous positions for loop detection
    
    def spawn_food(self):
        return (random.randrange(GAME_AREA_X, GAME_AREA_X + GAME_AREA_WIDTH, CELL_SIZE),
                random.randrange(GAME_AREA_Y, GAME_AREA_Y + GAME_AREA_HEIGHT, CELL_SIZE))
    
    def detect_loop(self):
        if len(self.previous_positions) > 10:
            last_positions = self.previous_positions[-10:]
            if len(set(last_positions)) < 5:
                return True
        return False
    
    def choose_direction(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        best_move = None
        min_distance = float("inf")
        for move in DIRECTIONS:
            new_x, new_y = head_x + move[0] * CELL_SIZE, head_y + move[1] * CELL_SIZE
            if (new_x, new_y) in self.snake or not (GAME_AREA_X <= new_x < GAME_AREA_X + GAME_AREA_WIDTH and GAME_AREA_Y <= new_y < GAME_AREA_Y + GAME_AREA_HEIGHT):
                continue  # Skip unsafe moves
            distance = abs(food_x - new_x) + abs(food_y - new_y)
            if distance < min_distance:
                min_distance = distance
                best_move = move
        return best_move if best_move else random.choice(DIRECTIONS)
    
    def move(self):
        if not self.alive:
            return
        self.direction = self.choose_direction()
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0] * CELL_SIZE, head_y + self.direction[1] * CELL_SIZE)
        
        if new_head in self.snake or not (GAME_AREA_X <= new_head[0] < GAME_AREA_X + GAME_AREA_WIDTH and GAME_AREA_Y <= new_head[1] < GAME_AREA_Y + GAME_AREA_HEIGHT):
            self.alive = False
            return
        
        self.snake.insert(0, new_head)
        self.previous_positions.append(new_head)
        
        if self.detect_loop():
            self.score -= 2  # Penalize loop behavior
        
        if new_head == self.food:
            self.score += 10
            self.length += 1  # Increase length counter when food is eaten
            self.food = self.spawn_food()
        else:
            self.snake.pop()
        
        self.score += 0.5
        self.score -= 0.1

def draw_game():
    screen.fill(BACKGROUND_GRAY)
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH + SIDE_BAR_WIDTH, TOP_BAR_HEIGHT))
    score_text = font.render(f"Best Score: {max(snake.score for snake in snakes)}, Length: {max(snake.length for snake in snakes)}", True, WHITE)
    screen.blit(score_text, (20, 10))
    
    for snake in snakes:
        if snake.alive:
            for segment in snake.snake:
                pygame.draw.rect(screen, GREEN, (segment[0] + GAP, segment[1] + GAP, CELL_SIZE - GAP * 2, CELL_SIZE - GAP * 2), border_radius=5)
            pygame.draw.rect(screen, RED, (snake.food[0] + GAP, snake.food[1] + GAP, CELL_SIZE - GAP * 2, CELL_SIZE - GAP * 2), border_radius=5)
    
    pygame.display.flip()
    clock.tick(FPS)

# Run Simulation with Visualization
population_size = 50
snakes = [SnakeAI() for _ in range(population_size)]

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
            break
        
        for snake in alive_snakes:
            snake.move()
        
        draw_game()
    
    snakes.sort(key=lambda s: s.score, reverse=True)
    
    top_performers = snakes[:10]
    new_population = []
    for _ in range(population_size - 1):
        parent = random.choice(top_performers)
        offspring = SnakeAI()
        offspring.brain = parent.brain + np.random.randn(4) * 0.2
        new_population.append(offspring)
    
    new_population.append(top_performers[0])
    snakes = new_population

for generation in range(10):
    print(f"Generation {generation} - Best Score: {max(snake.score for snake in snakes)}, Length: {max(snake.length for snake in snakes)}")
    run_generation()