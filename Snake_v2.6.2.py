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
        self.previous_positions = []  # Store previous positions for loop detection
        self.visited_positions = set()  # Track visited positions for exploration reward
    
    def spawn_food(self):
        return (random.randrange(0, WIDTH, CELL_SIZE), random.randrange(0, HEIGHT, CELL_SIZE))
    
    def is_trapped(self):
        """Check if the snake is surrounded with very few escape routes."""
        head_x, head_y = self.snake[0]
        free_spaces = 0
        for move in DIRECTIONS:
            new_x, new_y = head_x + move[0] * CELL_SIZE, head_y + move[1] * CELL_SIZE
            if (0 <= new_x < WIDTH and 0 <= new_y < HEIGHT) and (new_x, new_y) not in self.snake:
                free_spaces += 1
        return free_spaces <= 1  # If only one free space, it's in danger
    
    def detect_loop(self):
        """Detect if the snake is repeating positions, leading to a loop."""
        if len(self.previous_positions) > 10:
            last_positions = self.previous_positions[-10:]
            if len(set(last_positions)) < 5:  # If fewer than 5 unique positions in last 10 moves
                return True
        return False
    
    def lookahead_penalty(self, move):
        """Check if a move leads to a dead-end in the next few steps."""
        head_x, head_y = self.snake[0]
        new_x, new_y = head_x + move[0] * CELL_SIZE, head_y + move[1] * CELL_SIZE
        free_spaces = 0
        for future_move in DIRECTIONS:
            fut_x, fut_y = new_x + future_move[0] * CELL_SIZE, new_y + future_move[1] * CELL_SIZE
            if (0 <= fut_x < WIDTH and 0 <= fut_y < HEIGHT) and (fut_x, fut_y) not in self.snake:
                free_spaces += 1
        return -free_spaces * 2  # Higher penalty if fewer escape routes remain
    
    def choose_direction(self):
        if self.is_trapped():  # Escape mode
            return random.choice(DIRECTIONS)
        
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        best_move = None
        min_distance = float("inf")
        best_score = float("-inf")
        for move in DIRECTIONS:
            new_x, new_y = head_x + move[0] * CELL_SIZE, head_y + move[1] * CELL_SIZE
            if (new_x, new_y) in self.snake or not (0 <= new_x < WIDTH and 0 <= new_y < HEIGHT):
                continue  # Skip unsafe moves
            distance = abs(food_x - new_x) + abs(food_y - new_y)
            exploration_bonus = 1 if (new_x, new_y) not in self.visited_positions else 0  # Reduced exploration bonus
            move_score = -distance + self.lookahead_penalty(move) + exploration_bonus  # Consider distance, lookahead, and exploration
            if move_score > best_score:
                best_score = move_score
                best_move = move
        return best_move if best_move else random.choice(DIRECTIONS)  # Default to any move if no safe option
    
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
        self.previous_positions.append(new_head)  # Track movement history
        self.visited_positions.add(new_head)  # Track visited positions
        
        # Loop Detection Penalty
        if self.detect_loop():
            self.score -= 2  # Reduce loop penalty to encourage adaptation
        
        # Check for food
        if new_head == self.food:
            self.score += 12  # Increased food reward
            self.food = self.spawn_food()
        else:
            self.snake.pop()
        
        # Adjust efficiency penalty
        self.score -= 0.05 if (new_head in self.visited_positions) else 0.1  # Less penalty if exploring
        self.score += 0.5  # Small survival reward
    
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
    
    # Crossover & Mutation - Increased Mutation Strength
    top_performers = snakes[:10]
    new_population = []
    for _ in range(population_size - 1):  # Keep one elite
        parent = random.choice(top_performers)
        offspring = SnakeAI()
        offspring.brain = parent.brain + np.random.randn(4) * 0.2  # Increased mutation strength
        new_population.append(offspring)
    
    new_population.append(top_performers[0])  # Keep best performer
    snakes = new_population

# Evolve over generations with visualization
for generation in range(10):
    print(f"Generation {generation} - Best Score: {max(snake.score for snake in snakes)}")
    run_generation()
