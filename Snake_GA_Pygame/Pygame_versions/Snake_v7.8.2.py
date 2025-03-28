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
PADDING = 0
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
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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
best_score_overall = 0
best_length_overall = 0

# AI-Controlled Snake Class
class SnakeAI:
    def __init__(self, brain=None):
        self.snake = [(GAME_AREA_X + GAME_AREA_WIDTH // 2, GAME_AREA_Y + GAME_AREA_HEIGHT // 2)]
        self.food = self.spawn_food()
        self.direction = random.choice(DIRECTIONS)
        self.moves_made = 0 # Track the total moves the snake makes
        self.score = 0 # Score of the snake
        self.fitness_score = 0 # Fitness score of the snake
        self.length = 0    # Length of the snake
        self.alive = True
        self.start_time = time.time()
        self.last_food_time = time.time() # Time when the last food was collected

        # AI Weights
        self.brain = brain if brain is not None else np.random.rand(9) * 2 - 1  # Random weights between -1 and 1
        self.previous_positions = []

    def spawn_food(self):
        while True:
            food_x = random.randrange(GAME_AREA_X, GAME_AREA_X + GAME_AREA_WIDTH, CELL_SIZE)
            food_y = random.randrange(GAME_AREA_Y, GAME_AREA_Y + GAME_AREA_HEIGHT, CELL_SIZE)
            if (food_x, food_y) not in self.snake:
                return food_x, food_y

    def detect_loop(self):
        """ Detects if the snake is stuck in a repetitive loop and kills it if repeated 3 times. """
        loop_length = 10  # Define how many moves count as a potential loop
        if len(self.previous_positions) < loop_length * 3:
            return False  # Not enough moves to check for three repetitions
        
        # Extract the last three potential loops
        last_loop = self.previous_positions[-loop_length:]
        second_last_loop = self.previous_positions[-loop_length*2:-loop_length]
        third_last_loop = self.previous_positions[-loop_length*3:-loop_length*2]

        # If the last three loops are identical, return True (loop detected)
        if last_loop == second_last_loop == third_last_loop:
            return True

        return False


    def fitness_function(self):
        """
        Evaluates the fitness of a snake based on its score, survival time, and efficiency in collecting food.
        Penalizes excessive moves per food collection.
        """

        # Formula: (score^2) - (moves_made / (score + 1))
        fitness = (self.score * self.score) - (self.moves_made / max(1, self.score + 1))

        return fitness


    def choose_direction(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        move_scores = []
        valid_moves = []

        for move in DIRECTIONS:
            new_x, new_y = head_x + move[0] * CELL_SIZE, head_y + move[1] * CELL_SIZE
            if (new_x, new_y) in self.snake or not (GAME_AREA_X <= new_x < GAME_AREA_X + GAME_AREA_WIDTH and GAME_AREA_Y <= new_y < GAME_AREA_Y + GAME_AREA_HEIGHT):
                continue

            distance = abs(food_x - new_x) + abs(food_y - new_y)
            food_reward = self.brain[0] * (100 if (new_x, new_y) == self.food else 0)
            toward_food_reward = self.brain[1] * -distance
            away_food_penalty = self.brain[2] * (distance * 0.5)
            loop_penalty = self.brain[3] * (-10 if self.previous_positions.count((new_x, new_y)) > 2 else 0)
            survival_bonus = self.brain[4] * np.log1p(time.time() - self.start_time) * 3

           # Check if the new position is near a wall
            is_near_wall = new_x < CELL_SIZE or new_x > WIDTH - CELL_SIZE or new_y < CELL_SIZE or new_y > HEIGHT - CELL_SIZE

            # Check if food is near a wall
            is_food_near_wall = food_x < CELL_SIZE or food_x > WIDTH - CELL_SIZE or food_y < CELL_SIZE or food_y > HEIGHT - CELL_SIZE

            # Apply a small penalty if near a wall (but not if food is near)
            wall_penalty = self.brain[5] * (-3 if is_near_wall and not is_food_near_wall else 0)

            # **Directional Momentum:** Reward continuing in the same direction
            momentum_bonus = self.brain[7] * (10 if move == self.direction else 0)
            
            # Exploration bonus (Encourages visiting new positions instead of repeating)
            unique_positions = len(set(self.previous_positions))
            exploration_bonus = self.brain[6] * (unique_positions / (len(self.previous_positions) + 1))

            # Self-collision avoidance with lookahead
            lookahead_positions = [(new_x + dx * CELL_SIZE, new_y + dy * CELL_SIZE) for dx, dy in DIRECTIONS]
            lookahead_penalty = -5 if sum(pos in self.snake for pos in lookahead_positions) >= 2 else 0

            # Dead end detection
            lookahead_count = sum(
                (nx, ny) not in self.snake and 0 <= nx < WIDTH and 0 <= ny < HEIGHT
                for nx, ny in [(new_x + dx * CELL_SIZE, new_y + dy * CELL_SIZE) for dx, dy in DIRECTIONS]
            )
            dead_end_penalty = self.brain[8] * (-20 if lookahead_count <= 1 else 0)

            total_score = (
                food_reward + toward_food_reward - away_food_penalty +
                loop_penalty + survival_bonus - wall_penalty +
                exploration_bonus + lookahead_penalty + momentum_bonus + dead_end_penalty
            )
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

        # Collision check
        if new_head in self.snake or not (GAME_AREA_X <= new_head[0] < GAME_AREA_X + GAME_AREA_WIDTH 
                                        and GAME_AREA_Y <= new_head[1] < GAME_AREA_Y + GAME_AREA_HEIGHT):
            self.alive = False
            return

        # Move the snake
        self.snake.insert(0, new_head)
        self.previous_positions.append(new_head)
        
        self.moves_made += 1  # Track the total moves the snake makes


        # Check loop detection
        if self.detect_loop():
            self.alive = False 

        # Check for food collection
        if new_head == self.food:
            self.score += 20  
            self.length += 1
            self.food = self.spawn_food()
        else:
            self.snake.pop()  # Move the snake

        # Add survival bonus
        self.score += 1 + (self.length * 0.5)

        # Kill if survival time exceeds limit with low length
        if time.time() - self.start_time > 10 and self.length < 20:
            self.alive = False

        # **Update fitness**
        self.fitness_score = self.fitness_function()


def draw_game():
    screen.fill(BACKGROUND_GRAY)
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH + SIDE_BAR_WIDTH, TOP_BAR_HEIGHT))
    
    best_score = max((snake.score for snake in snakes if snake.alive), default=0)
    best_length = max((snake.length for snake in snakes if snake.alive), default=0)
    elapsed_time = round(time.time() - generation_start_time, 2)
    
    score_text = font.render(f"Best Score: {best_score}, Length: {best_length}, Time: {elapsed_time}s", True, WHITE)
    screen.blit(score_text, (20, 10))

    for snake in snakes:
        if snake.alive:
            for segment in snake.snake:
                pygame.draw.rect(screen, GREEN, (segment[0], segment[1], CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, RED, (snake.food[0], snake.food[1], CELL_SIZE, CELL_SIZE))
    
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
            break  # Stop the generation if all snakes are dead

        for snake in alive_snakes:
            snake.move()

        draw_game()

    # **Sort snakes by fitness**
    snakes.sort(key=lambda s: s.fitness_function(), reverse=True)  

    # **Print best snake fitness for debugging**
    best_fitness = snakes[0].fitness_function()  
    print(f"Best Fitness This Generation: {best_fitness}")

    # **Evolve new snakes using genetic algorithm**
    snakes = evolve_snakes(snakes)

def evolve_snakes(snakes):
    """ Evolves the population by selecting top performers, mutating, and generating offspring """
    top_performers = snakes[:10]  # Select the top 10 fittest snakes
    new_snakes = []

    for _ in range(len(snakes)):  # Maintain same population size
        parent1 = random.choice(top_performers)
        parent2 = random.choice(top_performers)

        # **Crossover: Mix weights from two parents**
        new_brain = (parent1.brain + parent2.brain) / 2  

        # **Mutation: Apply small random variation**
        mutation = np.random.randn(len(new_brain)) * 0.1  
        new_brain += mutation

        # **Create new snake with evolved brain**
        new_snakes.append(SnakeAI(brain=new_brain))

    return new_snakes


snakes = [SnakeAI() for _ in range(50)]

for generation in range(10):
    best_score = max((snake.score for snake in snakes if snake.alive), default=0)
    best_length = max((snake.length for snake in snakes if snake.alive), default=0)
    elapsed_time = round(time.time() - generation_start_time, 2)
    
    print(f"Generation {generation} - Best Score: {best_score}, Length: {best_length}, Time: {elapsed_time}s")
    run_generation()