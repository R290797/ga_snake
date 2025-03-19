from Menu import menu_screen
from Manual_gameplay import run_manual_mode, ManualKeysSnake
import pygame
import random
import numpy as np
import time
import sys  


# Initialize Pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 600, 600
TOP_BAR_HEIGHT = 50
SIDE_BAR_WIDTH = 200
PADDING = 20
CELL_SIZE = 20
GAP = 2
FPS = 60

# Colors
WHITE = (255, 255, 255)
BACKGROUND_GRAY = (215,210,203)
GRAY = (100, 100, 100)
GREEN = (21, 71, 52)
RED = (128,5,0)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)

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
# Track learning metrics across generations
generation_fitness = []
generation_avg_fitness = []
generation_lengths = []
generation_avg_lengths = []


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

        self.border_walls = set()
        for x in range(GAME_AREA_X - CELL_SIZE, GAME_AREA_X + GAME_AREA_WIDTH + CELL_SIZE, CELL_SIZE):
            self.border_walls.add((x, GAME_AREA_Y - CELL_SIZE))  # Top border
            self.border_walls.add((x, GAME_AREA_Y + GAME_AREA_HEIGHT))  # Bottom border

        for y in range(GAME_AREA_Y - CELL_SIZE, GAME_AREA_Y + GAME_AREA_HEIGHT + CELL_SIZE, CELL_SIZE):
            self.border_walls.add((GAME_AREA_X - CELL_SIZE, y))  # Left border
            self.border_walls.add((GAME_AREA_X + GAME_AREA_WIDTH, y))  # Right border

    
    def spawn_food(self):
        while True:
            food_x = random.randrange(GAME_AREA_X, GAME_AREA_X + GAME_AREA_WIDTH, CELL_SIZE)
            food_y = random.randrange(GAME_AREA_Y, GAME_AREA_Y + GAME_AREA_HEIGHT, CELL_SIZE)
            if (food_x, food_y) not in self.snake:
                return food_x, food_y

    def detect_loop(self):
        """ Detects if the snake is stuck in a repetitive loop and dynamically adjusts penalties. """
        loop_length = max(6, min(12, self.length // 2))  # Dynamic loop length based on snake size
        history_window = max(15, self.length * 2)  # Adjust based on snake length

        if len(self.previous_positions) < history_window:
            return False  # Not enough data to detect a loop

        # Count occurrences of each position in the last `history_window` moves
        position_counts = {}
        for pos in self.previous_positions[-history_window:]:
            position_counts[pos] = position_counts.get(pos, 0) + 1

        max_repeats = max(position_counts.values())

        # Dynamically determine loop threshold
        loop_threshold = 3 if self.length < 10 else 4  # Stricter for smaller snakes

        # If a position is visited too many times, it suggests a loop
        if max_repeats >= loop_threshold:
            return True

        return False




    def fitness_function(self):
        """
        Evaluates the fitness of a snake based on its score, survival time, and efficiency in collecting food.
        Penalizes excessive moves per food collection.
        """

        # Formula: (score^2) - (moves_made / (score + 1))
        fitness = (self.score * self.length * 2) + (time.time() - self.start_time) * 5 - (self.moves_made / 10)

        return fitness


    def choose_direction(self):
        def simulate_move(head, direction, depth=0):
            """Simulates a move and returns a heuristic score based on the resulting state."""
            new_x, new_y = head[0] + direction[0] * CELL_SIZE, head[1] + direction[1] * CELL_SIZE

            # Collision or wall check
            if (new_x, new_y) in self.snake or (new_x, new_y) in self.border_walls:
                return -1000  # Heavy penalty for moving into a wall or itself

            # Calculate distance to food
            distance_to_food = abs(self.food[0] - new_x) + abs(self.food[1] - new_y)

            #  Reduce food bonus (75 instead of 100) to prevent overriding safety concerns**
            food_bonus = self.brain[0] * (100 if (new_x, new_y) == self.food else 0)

            #  Remove conflicting away-food penalty**
            toward_food_reward = self.brain[1] * -distance_to_food

            #  Make loop penalty progressive (-10 per visit, up to -30 max)**
            visit_count = self.previous_positions.count((new_x, new_y))
            loop_penalty = self.brain[3] * (-20 * visit_count if visit_count > 1 else 0)

            #  Increase wall penalty (-5 instead of -2) for better obstacle avoidance**
            is_near_wall = new_x < CELL_SIZE or new_x > WIDTH - CELL_SIZE or new_y < CELL_SIZE or new_y > HEIGHT - CELL_SIZE
            is_food_near_wall = self.food[0] < CELL_SIZE or self.food[0] > WIDTH - CELL_SIZE or self.food[1] < CELL_SIZE or self.food[1] > HEIGHT - CELL_SIZE
            wall_penalty = self.brain[5] * (-3 if is_near_wall and not is_food_near_wall else 0)

            # Reduce momentum bonus (5 instead of 10) & make it progressive
            recent_direction = self.previous_positions[-10:] # Check last 5 moves
            same_direction_count = sum(1 for pos in recent_direction if pos == self.direction)
            momentum_bonus = self.brain[7] * (10 if same_direction_count >= 3 else 0)

            #   Apply exponential decay to exploration bonus (more important early-game)
            unique_positions = len(set(self.previous_positions))
            exploration_bonus = self.brain[6] * (unique_positions / (len(self.previous_positions) + 1)) * np.exp(-0.05 * len(self.previous_positions))

            #   Merge dead-end detection with self-collision penalty
            lookahead_positions = [(new_x + dx * CELL_SIZE, new_y + dy * CELL_SIZE) for dx, dy in DIRECTIONS]
            lookahead_collisions = sum(pos in self.snake for pos in lookahead_positions)
            dead_end_penalty = self.brain[8] * (-20 if lookahead_collisions >= 2 else 0)

            #  Improve recursive lookahead to break loops**
            if depth > 0 and visit_count > 1:
                loop_penalty -= 10 * depth  # Increase penalty deeper in recursion

            # Compute total score
            total_score = (
                food_bonus + toward_food_reward +
                loop_penalty + wall_penalty +
                exploration_bonus + momentum_bonus + dead_end_penalty
            )

            #*Recursive Lookahead (Modified)
            if depth < 2:  # Look ahead up to 3 moves
                future_scores = [simulate_move((new_x, new_y), next_move, depth + 1) for next_move in DIRECTIONS]
                best_future_score = max(future_scores)

                #  If future best move still results in a loop, force a different choice
                if best_future_score < -50:
                    total_score -= 30  # Stronger loop deterrent
                else:
                    total_score += best_future_score * 0.7  # Discounted future rewards

            return total_score

        #Evaluate All Possible First Moves
        head_x, head_y = self.snake[0]
        best_move = None
        best_score = float('-inf')

        for move in DIRECTIONS:
            score = simulate_move((head_x, head_y), move)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_move else random.choice(DIRECTIONS)  # Fallback to random if no move is found



    def move(self):
        if not self.alive:
            return
        
        self.direction = self.choose_direction()
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0] * CELL_SIZE, head_y + self.direction[1] * CELL_SIZE)

        # Collision check
        if new_head in self.snake or new_head in self.border_walls:
            self.alive = False
            return

        # Move the snake
        self.snake.insert(0, new_head)
        self.previous_positions.append(new_head)
        
        self.moves_made += 1  # Track the total moves the snake makes


        # Check loop detection
        if time.time() - self.last_food_time > 15:  # Extend starvation time
            self.alive = False  

        if self.detect_loop():
            self.fitness_score -= 50  # Penalize fitness score
            if random.random() > 0.5:  # 50% chance to survive the loop
                self.alive = False  



        # Check for food collection
        if new_head == self.food:
            self.score += 50
            self.length += 1
            self.food = self.spawn_food()
            self.last_food_time = time.time()  # Update time when food is eaten
        else:
            self.snake.pop()  # Move the snake

        # Add survival bonus
        self.score += 1 + (self.length * 2.5)

        if time.time() - self.last_food_time > 10 and self.length < 500:
            self.alive = False  # Kill snake if no food eaten in 10 seconds


        # **Update fitness**
        self.fitness_score = self.fitness_function()

def draw_snake_length_visualization():
    """Displays the length of the best snake visually on the right side of the screen."""
    bar_x = WIDTH + 50  # Position to the right of the game area
    bar_y = 100  # Start position for visualization
    bar_width = 20  # Width of each unit in visualization
    unit_height = 10  # Height of each unit
    spacing = 2  # Space between units

    best_length = max((snake.length for snake in snakes if snake.alive), default=0)

    for i in range(best_length):
        pygame.draw.rect(screen, GREEN, (bar_x, bar_y + i * (unit_height + spacing), bar_width, unit_height))


def draw_game():
    screen.fill(BACKGROUND_GRAY)
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH + SIDE_BAR_WIDTH, TOP_BAR_HEIGHT))

    best_score = max((snake.score for snake in snakes if snake.alive), default=0)
    best_length = max((snake.length for snake in snakes if snake.alive), default=0)
    avg_length = np.mean([snake.length for snake in snakes]) if snakes else 0
    elapsed_time = round(time.time() - generation_start_time, 2)

    # **Divide Top Bar into 5 Equal Sections**
    section_width = WIDTH // 4  # Divide top bar into 5 sections

    # **Draw Dividers**
    for i in range(1, 4):  # Create 4 vertical lines to divide sections
        pygame.draw.line(screen, WHITE, (i * section_width, 0), (i * section_width, TOP_BAR_HEIGHT), 2)

    # **Display Information in Each Section**
    font = pygame.font.SysFont(None, 30)
    score_text = font.render(f"Score: {best_score}", True, WHITE)
    time_text = font.render(f"Time: {elapsed_time}s", True, WHITE)
    length_text = font.render(f"Length: {best_length}", True, WHITE)
    avg_length_text = font.render(f"Avg Len: {avg_length:.2f}", True, WHITE)

    # **Position Each Text in its Section**
    screen.blit(score_text, (section_width * 3.1, 10))  # First section
    screen.blit(time_text, (section_width * 0.1, 10))   # Second section
    screen.blit(length_text, (section_width * 1.1, 10)) # Third section
    screen.blit(avg_length_text, (section_width * 2.1, 10))   # Fifth section

    # **Display the best lengths for the last 10 generations in the right space**
    text_x = WIDTH + 30  # Position on the right side
    text_y = 100  # Start position

    gen_text = font.render("Generations", True, BLACK)
    screen.blit(gen_text, (text_x, text_y))

    for i, length in enumerate(generation_lengths):
        length_text = font.render(f"- Gen {i}: {length}", True, GREEN)
        screen.blit(length_text, (text_x, text_y + (i + 1) * 30))  # Spacing between lines

    # **Draw Borders One Grid Outside**
    for x in range(GAME_AREA_X - CELL_SIZE, GAME_AREA_X + GAME_AREA_WIDTH + CELL_SIZE, CELL_SIZE):
        pygame.draw.rect(screen, BROWN, (x, GAME_AREA_Y - CELL_SIZE, CELL_SIZE, CELL_SIZE), border_radius=5)
        pygame.draw.rect(screen, BROWN, (x, GAME_AREA_Y + GAME_AREA_HEIGHT, CELL_SIZE, CELL_SIZE), border_radius=5)

    for y in range(GAME_AREA_Y - CELL_SIZE, GAME_AREA_Y + GAME_AREA_HEIGHT + CELL_SIZE, CELL_SIZE):
        pygame.draw.rect(screen, BROWN, (GAME_AREA_X - CELL_SIZE, y, CELL_SIZE, CELL_SIZE), border_radius=5)
        pygame.draw.rect(screen, BROWN, (GAME_AREA_X + GAME_AREA_WIDTH, y, CELL_SIZE, CELL_SIZE), border_radius=5)

    # **Draw Snakes and Food**
    for snake in snakes:
        if snake.alive:
            for segment in snake.snake:
                pygame.draw.rect(screen, GREEN, (segment[0] + GAP, segment[1] + GAP, CELL_SIZE - GAP * 2, CELL_SIZE - GAP * 2), border_radius=5)
            pygame.draw.rect(screen, RED, (snake.food[0] + GAP, snake.food[1] + GAP, CELL_SIZE - GAP * 2, CELL_SIZE - GAP * 2), border_radius=5)
    
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

    # **Calculate Key Metrics**
    best_snake = max(snakes, key=lambda s: s.fitness_function(), default=None)
    best_fitness = best_snake.fitness_function() if best_snake else 0
    avg_fitness = np.mean([snake.fitness_function() for snake in snakes])

    best_length = max((snake.length for snake in snakes), default=0)
    avg_length = np.mean([snake.length for snake in snakes])

    # **Retrieve Best Snake's Weights (Brain Parameters)**
    best_weights = best_snake.brain if best_snake else np.zeros(9)

    # **Print Generation Summary**
    print("=" * 50)
    print(f" Generation {len(generation_fitness) + 1} Summary ")
    print("=" * 50)
    print(f" 🏆 Best Fitness Score: {best_fitness:.2f}")
    print(f" 📊 Average Fitness Score: {avg_fitness:.2f}")
    print(f" 🐍 Best Length Achieved: {best_length}")
    print(f" 📏 Average Length of Snakes: {avg_length:.2f}")
    print("-" * 50)
    print(" 🧠 Inherited Weights (Brain Parameters)")
    print(f"  - Food Bonus Weight: {best_weights[0]:.3f}")
    print(f"  - Toward Food Weight: {best_weights[1]:.3f}")
    print(f"  - Away Food Penalty: {best_weights[2]:.3f}")
    print(f"  - Loop Penalty: {best_weights[3]:.3f}")
    print(f"  - Survival Bonus: {best_weights[4]:.3f}")
    print(f"  - Wall Penalty: {best_weights[5]:.3f}")
    print(f"  - Exploration Bonus: {best_weights[6]:.3f}")
    print(f"  - Momentum Bonus: {best_weights[7]:.3f}")
    print(f"  - Dead-End Penalty: {best_weights[8]:.3f}")
    print("=" * 50)

    # **Store Data for Future Analysis**
    generation_fitness.append(best_fitness)
    generation_avg_fitness.append(avg_fitness)
    generation_lengths.append(best_length)

    # **Evolve Snakes for Next Generation**
    snakes = evolve_snakes(snakes)


def evolve_snakes(snakes):
    """ Evolves the population by selecting top performers, mutating, and generating offspring """
    top_performers = snakes[:10] # Select the top 5fittest snakes
    winner_snake = top_performers[0]
    new_snakes = []

    for _ in range(len(snakes)):
        parent1 = random.choice(top_performers[:3])  # Top 5 performers
        parent2 = random.choice(top_performers[3:])  # Mid-tier performers
        parent3 = random.choice(snakes)

        #Crossover: Mix weights from two parents
        new_brain = (parent1.brain + parent2.brain + parent3.brain ) / 3  

        #Mutation
        if len(generation_fitness) > 1 and generation_fitness[-1] > generation_fitness[-2]:
            mutation = np.random.randn(len(new_brain)) * 0.1  # Less mutation if improving
        else:
            mutation = np.random.randn(len(new_brain)) * 0.3  # More mutation if not improving
         
        new_brain += mutation

        # **Create new snake with evolved brain**
        new_snakes.append(SnakeAI(brain=new_brain))

    return new_snakes


snakes = [SnakeAI() for _ in range(50)]

def show_game_over_screen(snake):
    screen.fill(BACKGROUND_GRAY)
    font_large = pygame.font.SysFont(None, 50)
    font_small = pygame.font.SysFont(None, 30)

    # **Game Over Message**
    game_over_text = font_large.render("Game Over", True, BLACK)
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 3))

    # **Display Final Score, Time, and Length**
    elapsed_time = round(time.time() - snake.start_time, 2)
    stats_text = [
        f"Time: {elapsed_time}s",
        f"Score: {snake.score}",
        f"Length: {snake.length}"
    ]

    for i, text in enumerate(stats_text):
        stat_render = font_small.render(text, True, BLACK)
        screen.blit(stat_render, (WIDTH // 2 - stat_render.get_width() // 2, HEIGHT // 2 + i * 40))

    # **Button Positions**
    button_width, button_height = 200, 50
    replay_button = pygame.Rect(WIDTH // 2 - button_width - 20, HEIGHT // 1.5, button_width, button_height)
    quit_button = pygame.Rect(WIDTH // 2 + 20, HEIGHT // 1.5, button_width, button_height)

    pygame.draw.rect(screen, GREEN, replay_button, border_radius=10)
    pygame.draw.rect(screen, RED, quit_button, border_radius=10)

    # **Render Button Text**
    replay_text = font_small.render("Replay", True, WHITE)
    quit_text = font_small.render("Quit", True, WHITE)

    screen.blit(replay_text, (replay_button.x + button_width // 2 - replay_text.get_width() // 2,
                              replay_button.y + button_height // 2 - replay_text.get_height() // 2))
    screen.blit(quit_text, (quit_button.x + button_width // 2 - quit_text.get_width() // 2,
                            quit_button.y + button_height // 2 - quit_text.get_height() // 2))

    pygame.display.flip()

    # **Wait for User Input**
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if replay_button.collidepoint(event.pos):
                    return "replay"  # Restart the game
                elif quit_button.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()


def run_manual_mode():
    while True:  # Allow replaying the game
        snake = ManualKeysSnake()

        while snake.alive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # Handle Key Press for Direction Change
                if event.type == pygame.KEYDOWN and event.key in DIRECTIONS:
                    new_direction = DIRECTIONS[event.key]
                    if (new_direction[0] * -1, new_direction[1] * -1) != snake.direction:  # Prevent reversing
                        snake.direction = new_direction

            snake.move()
            draw_game()
            clock.tick(FPS)

        # **Show Game Over Screen and Handle Replay**
        action = show_game_over_screen(snake)
        if action == "replay":
            continue  # Restart the loop to replay
        else:
            break  # Quit the game

def get_training_parameters():
    screen.fill(BACKGROUND_GRAY)
    font_large = pygame.font.SysFont(None, 50)
    font_small = pygame.font.SysFont(None, 30)

    # **Prompt User**
    title_text = font_large.render("AI Training Setup", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 4))

# **Adjusted Input Box Positions**
    input_boxes = {
        "snakes_per_gen": pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 40, 200, 40),  # Move down slightly
        "num_generations": pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 40, 200, 40),  # More spacing from first box
    }



    input_values = {"snakes_per_gen": "", "num_generations": ""}
    active_box = None

    # **Submit Button**
    submit_button = pygame.Rect(WIDTH // 2 - 50, HEIGHT // 2 + 80, 100, 40)

    
    while True:
        screen.fill(BACKGROUND_GRAY)
        screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 4))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Handle Clicks on Input Boxes
            if event.type == pygame.MOUSEBUTTONDOWN:
                for key, box in input_boxes.items():
                    if box.collidepoint(event.pos):
                        active_box = key
                if submit_button.collidepoint(event.pos):  # If submit button clicked
                    if input_values["snakes_per_gen"].isdigit() and input_values["num_generations"].isdigit():
                        return int(input_values["snakes_per_gen"]), int(input_values["num_generations"])

            # Handle Keyboard Input
            if event.type == pygame.KEYDOWN:
                if active_box:
                    if event.key == pygame.K_BACKSPACE:
                        input_values[active_box] = input_values[active_box][:-1]
                    elif event.key in range(pygame.K_0, pygame.K_9 + 1):  # Only allow numbers
                        input_values[active_box] += event.unicode

        # **Render Input Boxes**
        for key, box in input_boxes.items():
            pygame.draw.rect(screen, WHITE, box, border_radius=5)
            text_surface = font_small.render(input_values[key], True, BLACK)
            screen.blit(text_surface, (box.x + 10, box.y + 10))

# **Render Labels Centered Above Input Boxes**
        label_x_offset = 100  # Offset to center above input box
        screen.blit(font_small.render("# of Snakes:", True, BLACK), 
                    (input_boxes["snakes_per_gen"].x + label_x_offset - 60, input_boxes["snakes_per_gen"].y - 40))
        screen.blit(font_small.render("# of Generations:", True, BLACK), 
                    (input_boxes["num_generations"].x + label_x_offset - 85, input_boxes["num_generations"].y - 40))


        # **Render Submit Button**
        pygame.draw.rect(screen, GREEN, submit_button, border_radius=10)
        submit_text = font_small.render("Start", True, WHITE)
        screen.blit(submit_text, (submit_button.x + submit_button.width // 2 - submit_text.get_width() // 2,
                                  submit_button.y + submit_button.height // 2 - submit_text.get_height() // 2))

        pygame.display.flip()

def show_training_summary(best_score, best_length, training_time):
    screen.fill(BACKGROUND_GRAY)
    font_large = pygame.font.SysFont(None, 50)
    font_small = pygame.font.SysFont(None, 30)

    # **Display Training Complete Message**
    title_text = font_large.render("Training Complete", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 4))

    # **Display Final Training Results**
    stats_text = [
        f"Final Best Score: {best_score}",
        f"Final Best Length: {best_length}",
        f"Total Training Time: {round(training_time, 2)}s"
    ]

    for i, text in enumerate(stats_text):
        stat_render = font_small.render(text, True, BLACK)
        screen.blit(stat_render, (WIDTH // 2 - stat_render.get_width() // 2, HEIGHT // 2 + i * 40))

    # **Button Positions**
    button_width, button_height = 200, 50
    replay_button = pygame.Rect(WIDTH // 2 - button_width - 20, HEIGHT // 1.5, button_width, button_height)
    quit_button = pygame.Rect(WIDTH // 2 + 20, HEIGHT // 1.5, button_width, button_height)

    pygame.draw.rect(screen, GREEN, replay_button, border_radius=10)
    pygame.draw.rect(screen, RED, quit_button, border_radius=10)

    # **Render Button Text**
    replay_text = font_small.render("Replay", True, WHITE)
    quit_text = font_small.render("Quit", True, WHITE)

    screen.blit(replay_text, (replay_button.x + button_width // 2 - replay_text.get_width() // 2,
                              replay_button.y + button_height // 2 - replay_text.get_height() // 2))
    screen.blit(quit_text, (quit_button.x + button_width // 2 - quit_text.get_width() // 2,
                            quit_button.y + button_height // 2 - quit_text.get_height() // 2))

    pygame.display.flip()

    # **Wait for User Input**
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if replay_button.collidepoint(event.pos):
                    return "replay"  # Restart AI training
                elif quit_button.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()


def run_pretrained_ai():
    print("Pre-trained AI mode coming soon...")

# **Show the menu before running the game**
selection = menu_screen()

if selection == "manual":
    print("Starting Manual Play...")
    run_manual_mode()


elif selection == "train":
    while True:  # Allow replaying AI training
        snakes_per_gen, num_generations = get_training_parameters()
        print(f"Starting AI Training with {snakes_per_gen} snakes per generation for {num_generations} generations.")

        # Initialize Snakes
        snakes = [SnakeAI() for _ in range(snakes_per_gen)]
        training_start_time = time.time()

        for generation in range(num_generations):
            best_score = max((snake.score for snake in snakes if snake.alive), default=0)
            best_length = max((snake.length for snake in snakes if snake.alive), default=0)
            elapsed_time = round(time.time() - training_start_time, 2)

            print(f"Generation {generation} - Best Score: {best_score}, Length: {best_length}, Time: {elapsed_time}s")
            run_generation()

        # **Show Training Summary and Handle Replay**
        action = show_training_summary(best_score, best_length, round(time.time() - training_start_time, 2))
        if action == "replay":
            continue  # Restart training
        else:
            break  # Quit



elif selection == "pretrained":
    print("Running Pre-Trained AI...")
    run_pretrained_ai()  # Function to use a pre-trained AI model

elif selection == "quit":
    print("Exiting Game...")
    pygame.quit()
    sys.exit()
