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
PADDING = 0  # Align game area with borders
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

# Global Time Tracker for Generations
generation_start_time = time.time()

# AI-Controlled Snake Class
class SnakeAI:
    def __init__(self):
        self.snake = [(GAME_AREA_X + GAME_AREA_WIDTH // 2, GAME_AREA_Y + GAME_AREA_HEIGHT // 2)]
        self.food = self.spawn_food()
        self.direction = random.choice(DIRECTIONS)
        self.score = 0
        self.length = 0  # Tracks total food eaten
        self.alive = True
        
        # AI Weights: [Food Reward, Moving Toward Food, Moving Away Penalty, Loop Penalty, Survival Bonus, Wall Avoidance]
        self.brain = np.random.rand(6) * 2 - 1  # Random weights between -1 and 1
        
        self.previous_positions = []  # Store previous positions for loop detection
    
    def spawn_food(self):
        while True:
            food_x = random.randrange(GAME_AREA_X, GAME_AREA_X + GAME_AREA_WIDTH, CELL_SIZE)
            food_y = random.randrange(GAME_AREA_Y, GAME_AREA_Y + GAME_AREA_HEIGHT, CELL_SIZE)
            if (food_x, food_y) not in self.snake:
                return food_x, food_y
    
    def detect_loop(self):
        if len(self.previous_positions) > 10:
            last_positions = self.previous_positions[-10:]
            if len(set(last_positions)) < 5:
                return True
        return False
    
    def choose_direction(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        move_scores = []
        valid_moves = []
        
        for move in DIRECTIONS:
            new_x, new_y = head_x + move[0] * CELL_SIZE, head_y + move[1] * CELL_SIZE
            if (new_x, new_y) in self.snake or not (GAME_AREA_X <= new_x < GAME_AREA_X + GAME_AREA_WIDTH and GAME_AREA_Y <= new_y < GAME_AREA_Y + GAME_AREA_HEIGHT):
                continue  # Skip invalid moves
            
            distance = abs(food_x - new_x) + abs(food_y - new_y)
            food_reward = self.brain[0] * (100 if (new_x, new_y) == self.food else 0)  # Strong food incentive
            toward_food_reward = self.brain[1] * -distance
            away_food_penalty = self.brain[2] * (distance * 0.5)  # Moving away is penalized
            loop_penalty = self.brain[3] * (-2 if (new_x, new_y) in self.previous_positions else 0)  # Avoid repeating moves
            survival_bonus = self.brain[4] * 5.0  # Reward for staying alive
            wall_penalty = self.brain[5] * (1.0 / (min(new_x - GAME_AREA_X, GAME_AREA_X + GAME_AREA_WIDTH - new_x) + 1))  # Wall avoidance
            
            total_score = food_reward + toward_food_reward - away_food_penalty + loop_penalty + survival_bonus - wall_penalty
            move_scores.append(total_score)
            valid_moves.append(move)
        
        if valid_moves:
            best_index = np.argmax(move_scores)
            return valid_moves[best_index]
        else:
            return random.choice(DIRECTIONS)
    
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
            self.score -= 2  # Penalize looping
        
        if new_head == self.food:
            self.score += 20  # Strong reward for eating
            self.length += 1
            self.food = self.spawn_food()
        else:
            self.snake.pop()
        
        self.score += 1 + (self.length * 0.5)  # Reward survival

def draw_game():
    screen.fill(BACKGROUND_GRAY)
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH + SIDE_BAR_WIDTH, TOP_BAR_HEIGHT))
    
    # Draw each snake
    for snake in snakes:
        if snake.alive:
            for segment in snake.snake:
                pygame.draw.rect(screen, GREEN, (segment[0], segment[1], CELL_SIZE, CELL_SIZE), border_radius=5)
            pygame.draw.rect(screen, RED, (snake.food[0], snake.food[1], CELL_SIZE, CELL_SIZE), border_radius=5)
    
    pygame.display.flip()
    clock.tick(FPS)

def run_generation():
    global snakes, generation_start_time
    generation_start_time = time.time()
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

population_size = 50
snakes = [SnakeAI() for _ in range(population_size)]

for generation in range(10):
    print(f"Generation {generation} - Best Score: {max(snake.score for snake in snakes)}, Length: {max(snake.length for snake in snakes)}, Time: {round(time.time() - generation_start_time, 2)}s")
    run_generation()