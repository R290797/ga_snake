from Menu import menu_screen
from Manual_gameplay import run_manual_mode, ManualKeysSnake
import pygame
import random
import numpy as np
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')
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
BACKGROUND_GRAY = (215, 210, 203)
GRAY = (100, 100, 100)
GREEN = (21, 71, 52)
RED = (128, 5, 0)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)
BLUE = (0, 79, 152)

# Directions
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Screen Configurations
screen = pygame.display.set_mode(
    (WIDTH + SIDE_BAR_WIDTH, HEIGHT + TOP_BAR_HEIGHT))
pygame.display.set_caption("Gen Snake")
clock = pygame.time.Clock()

# Font setup
font = pygame.font.SysFont(None, 30)

background_image = pygame.image.load("Assets/background.webp")

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


# --- Configuration & Hyperparameters ---
CONFIG = {
    # Base lookahead depth (can be adaptive)
    "LOOKAHEAD_DEPTH_BASE": 2,
    "LOOKAHEAD_DEPTH_THRESHOLD": 10,    # Snake length threshold for adjusting depth
    "FOOD_BONUS_CONSTANT": 75,
    "MUTATION_LOW": 0.1,                # Mutation factor when generation is improving
    "MUTATION_HIGH": 0.3,               # Mutation factor when improvement stalls
    "ELITISM_COUNT": 3,                 # Number of top snakes carried directly over
    "TOURNAMENT_SIZE": 3,               # Number of participants for tournament selection
    "DIVERSITY_INJECTION_PROB": 0.05      # Chance to insert a completely random snake
}

PRETRAINED_MODELS = {
    "Hunter":       [4.583, 1.024, -0.472, 0.315, -2.127, -1.038, 1.143, -1.476, -1.018],
    "Strategist":   [3.472, 0.832, -0.754, 0.206, -2.473, -0.482, 0.583, -1.219, -1.452],
    "Explorer": 	[2.765, 0.746, -0.621, 0.381, -1.825, -0.752, 1.493, -0.975, -1.189],
    "Risk Taker": 	[5.218, 1.217, -0.347, 0.089, -0.493, -2.013, -0.472, -1.839, -2.047],
    "AI Mastery":   [4.590, 0.791, 5.276, -0.006, -2.184, -3.037, 2.219, -1.856, -0.787],
    }

# AI-Controlled Snake Class


class SnakeAI:
    def __init__(self, brain=None):
        # Initialize snake at center
        self.snake = [(GAME_AREA_X + GAME_AREA_WIDTH // 2,
                       GAME_AREA_Y + GAME_AREA_HEIGHT // 2)]
        self.direction = random.choice(DIRECTIONS)
        self.moves_made = 0
        self.score = 0
        self.fitness_score = 0
        self.length = 0
        self.alive = True
        self.start_time = time.time()
        self.last_food_time = time.time()
        self.previous_positions = []
        self.previous_directions = []

        # AI Weights
        if brain is None:
            # Smaller range for stability
            self.brain = np.random.uniform(-1.5, 1.5, 9)
            self.brain /= np.linalg.norm(self.brain)  # Keep values balanced
        else:
            self.brain = np.array(brain)  # Ensure it's a NumPy array

        self.border_walls = set()
        for x in range(GAME_AREA_X - CELL_SIZE, GAME_AREA_X + GAME_AREA_WIDTH + CELL_SIZE, CELL_SIZE):
            self.border_walls.add((x, GAME_AREA_Y - CELL_SIZE))  # Top border
            self.border_walls.add(
                (x, GAME_AREA_Y + GAME_AREA_HEIGHT))  # Bottom border

        for y in range(GAME_AREA_Y - CELL_SIZE, GAME_AREA_Y + GAME_AREA_HEIGHT + CELL_SIZE, CELL_SIZE):
            self.border_walls.add((GAME_AREA_X - CELL_SIZE, y))  # Left border
            self.border_walls.add(
                (GAME_AREA_X + GAME_AREA_WIDTH, y))  # Right border

        self.food = self.spawn_food()

    def spawn_food(self):
        while True:
            food_x = random.randrange(
                GAME_AREA_X, GAME_AREA_X + GAME_AREA_WIDTH, CELL_SIZE)
            food_y = random.randrange(
                GAME_AREA_Y, GAME_AREA_Y + GAME_AREA_HEIGHT, CELL_SIZE)
            if (food_x, food_y) not in self.snake:
                return food_x, food_y

    def get_lookahead_depth(self):
        # Adaptive lookahead: deeper when snake is small, shallower when large
        if self.length < CONFIG["LOOKAHEAD_DEPTH_THRESHOLD"]:
            return CONFIG["LOOKAHEAD_DEPTH_BASE"] + 1
        return CONFIG["LOOKAHEAD_DEPTH_BASE"]

    def detect_loop(self):
        history_window = max(15, self.length * 2)
        if len(self.previous_positions) < history_window:
            return False
        position_counts = {}
        for pos in self.previous_positions[-history_window:]:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        loop_threshold = 3 if self.length < 10 else 4
        return max(position_counts.values()) >= loop_threshold

    def fitness_function(self):
        fitness = (self.score * self.length * 5) + (time.time() -
                                                    self.start_time) * 10 - (self.moves_made / (self.score + 1))
        food_streak_bonus = (self.score ** 1.5) * 2  # Encourages streaks
        fitness += food_streak_bonus

        return fitness

    def choose_direction(self):
        def simulate_move(head, direction, depth=0):
            new_x = head[0] + direction[0] * CELL_SIZE
            new_y = head[1] + direction[1] * CELL_SIZE

            # Check for collision with snake or walls
            if (new_x, new_y) in self.snake or (new_x, new_y) in self.border_walls:
                return -1000

            # Heuristic evaluation:
            distance_to_food = abs(
                self.food[0] - new_x) + abs(self.food[1] - new_y)
            food_bonus = self.brain[0] * (CONFIG["FOOD_BONUS_CONSTANT"]
                                          if (new_x, new_y) == self.food else 0)
            toward_food_reward = self.brain[1] * (-distance_to_food)
            visit_count = self.previous_positions.count((new_x, new_y))
            loop_penalty = self.brain[3] * \
                (-20 * visit_count if visit_count > 1 else 0)

            is_near_wall = new_x < CELL_SIZE or new_x > WIDTH - \
                CELL_SIZE or new_y < CELL_SIZE or new_y > HEIGHT - CELL_SIZE
            is_food_near_wall = self.food[0] < CELL_SIZE or self.food[0] > WIDTH - \
                CELL_SIZE or self.food[1] < CELL_SIZE or self.food[1] > HEIGHT - CELL_SIZE
            wall_penalty = self.brain[5] * \
                (-3 if is_near_wall and not is_food_near_wall else 0)

            recent_directions = self.previous_directions[-10:]
            same_direction_count = sum(
                1 for d in recent_directions if d == self.direction)
            momentum_bonus = self.brain[7] * \
                (10 if same_direction_count >= 3 else 0)

            unique_positions = len(set(self.previous_positions))
            exploration_bonus = self.brain[6] * (unique_positions / (
                len(self.previous_positions) + 1)) * np.exp(-0.05 * len(self.previous_positions))
            lookahead_positions = [
                (new_x + dx * CELL_SIZE, new_y + dy * CELL_SIZE) for dx, dy in DIRECTIONS]
            lookahead_collisions = sum(
                1 for pos in lookahead_positions if pos in self.snake)
            dead_end_penalty = self.brain[8] * \
                (-20 if lookahead_collisions >= 2 else 0)

            if depth > 0 and visit_count > 1:
                loop_penalty -= 10 * depth

            total_score = (food_bonus + toward_food_reward + loop_penalty +
                           wall_penalty + exploration_bonus + momentum_bonus + dead_end_penalty)

            # Recursive lookahead with adaptive depth:
            if depth < self.get_lookahead_depth():
                future_scores = [simulate_move(
                    (new_x, new_y), next_move, depth + 1) for next_move in DIRECTIONS]
                best_future_score = max(future_scores)
                total_score += best_future_score * 0.7 if best_future_score >= -50 else -30

            return total_score

        head = self.snake[0]
        best_move = None
        best_score = float('-inf')
        for move in DIRECTIONS:
            score = simulate_move(head, move)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move if best_move is not None else random.choice(DIRECTIONS)


    def move(self):
        if not self.alive:
            return

        self.direction = self.choose_direction()
        head_x, head_y = self.snake[0]
        new_head = (
            head_x + self.direction[0] * CELL_SIZE, head_y + self.direction[1] * CELL_SIZE)

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

    best_length = max(
        (snake.length for snake in snakes if snake.alive), default=0)

    for i in range(best_length):
        pygame.draw.rect(screen, GREEN, (bar_x, bar_y + i *
                         (unit_height + spacing), bar_width, unit_height))


def draw_game(game_mode="train_ai", model_params=None):
    screen.fill(BACKGROUND_GRAY)
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH +
                     SIDE_BAR_WIDTH, TOP_BAR_HEIGHT))

    best_score = max(
        (snake.score for snake in snakes if snake.alive), default=0)
    best_length = max(
        (snake.length for snake in snakes if snake.alive), default=0)
    avg_length = np.mean([snake.length for snake in snakes]) if snakes else 0
    elapsed_time = round(time.time() - generation_start_time, 2)

    # **Divide Top Bar into 5 Equal Sections**
    section_width = WIDTH // 4  # Divide top bar into 5 sections

    # **Draw Dividers**
    for i in range(1, 4):  # Create 4 vertical lines to divide sections
        pygame.draw.line(screen, WHITE, (i * section_width, 0),
                         (i * section_width, TOP_BAR_HEIGHT), 2)

    # **Display Information in Each Section**
    font = pygame.font.SysFont(None, 30)
    score_text = font.render(f"Score: {best_score}", True, WHITE)
    time_text = font.render(f"Time: {elapsed_time}s", True, WHITE)
    length_text = font.render(f"Length: {best_length}", True, WHITE)
    avg_length_text = font.render(f"Avg Len: {avg_length:.2f}", True, WHITE)

    # **Position Each Text in its Section**
    screen.blit(score_text, (section_width * 3.1, 10))  # First section
    screen.blit(time_text, (section_width * 0.1, 10))   # Second section
    screen.blit(length_text, (section_width * 1.1, 10))  # Third section
    screen.blit(avg_length_text, (section_width * 2.1, 10))   # Fifth section

    if game_mode == "train_ai":
        # **Display the best lengths for the last 10 generations in the right space**
        text_x = WIDTH + 30  # Position on the right side
        text_y = 100  # Start position

        gen_text = font.render("Generations", True, BLACK)
        screen.blit(gen_text, (text_x, text_y))

        for i, length in enumerate(generation_lengths):
            length_text = font.render(f"- Gen {i}: {length}", True, GREEN)
            # Spacing between lines)
            screen.blit(length_text, (text_x, text_y + (i + 1) * 30))

    elif game_mode == "pretrained_ai":
        # **Display the Pre-Trained Model Parameters**
        text_x = WIDTH + 2  # Position on the right side
        text_y = 100  # Start position

        # Reduce font size for better fit
        small_font = pygame.font.SysFont(None, 24)
        param_text = small_font.render("Parameter Weights", True, BLACK)

        screen.blit(param_text, (text_x, text_y))

        param_labels = [
            "- Food Bonus", "- Toward Food", "- Away Penalty", "- Loop Penalty",
            "- Survival Bonus", "- Wall Penalty", "- Exploration Bonus",
            "- Momentum Bonus", "- Dead-End Penalty"
        ]

        # Display each parameter with its value
        for i, (label, value) in enumerate(zip(param_labels, model_params)):
            param_value_text = small_font.render(
                f"{label}: {value:.2f}", True, GREEN)
            # Reduce spacing for better fit
            screen.blit(param_value_text, (text_x, text_y + (i + 1) * 22))

    # **Draw Borders One Grid Outside**
    for x in range(GAME_AREA_X - CELL_SIZE, GAME_AREA_X + GAME_AREA_WIDTH + CELL_SIZE, CELL_SIZE):
        pygame.draw.rect(screen, BROWN, (x, GAME_AREA_Y -
                         CELL_SIZE, CELL_SIZE, CELL_SIZE), border_radius=5)
        pygame.draw.rect(screen, BROWN, (x, GAME_AREA_Y +
                         GAME_AREA_HEIGHT, CELL_SIZE, CELL_SIZE), border_radius=5)

    for y in range(GAME_AREA_Y - CELL_SIZE, GAME_AREA_Y + GAME_AREA_HEIGHT + CELL_SIZE, CELL_SIZE):
        pygame.draw.rect(screen, BROWN, (GAME_AREA_X - CELL_SIZE,
                         y, CELL_SIZE, CELL_SIZE), border_radius=5)
        pygame.draw.rect(screen, BROWN, (GAME_AREA_X + GAME_AREA_WIDTH,
                         y, CELL_SIZE, CELL_SIZE), border_radius=5)

    # **Draw Snakes and Food**
    for snake in snakes:
        if snake.alive:
            for segment in snake.snake:
                pygame.draw.rect(
                    screen, GREEN, (segment[0] + GAP, segment[1] + GAP, CELL_SIZE - GAP * 2, CELL_SIZE - GAP * 2), border_radius=5)
            pygame.draw.rect(screen, RED, (snake.food[0] + GAP, snake.food[1] +
                             GAP, CELL_SIZE - GAP * 2, CELL_SIZE - GAP * 2), border_radius=5)

    pygame.display.flip()
    clock.tick(FPS)


def run_generation():
    global snakes, generation_start_time, best_score_overall, best_length_overall
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
    if best_snake is not None:
        # If this generation's snake is better, update overall
        if best_snake.score > best_score_overall:
            best_score_overall = best_snake.score
        if best_snake.length > best_length_overall:
            best_length_overall = best_snake.length
    best_fitness = best_snake.fitness_function() if best_snake else 0
    avg_fitness = np.mean([snake.fitness_function() for snake in snakes])

    best_length = max((snake.length for snake in snakes), default=0)
    avg_length = np.mean([snake.length for snake in snakes])

    # **Retrieve Best Snake's Weights (Brain Parameters)**
    best_weights = best_snake.brain if best_snake else np.zeros(9)

    # **Print Generation Summary**
    log_and_print("=" * 50)
    log_and_print(f" Generation {len(generation_fitness) + 1} Summary ")
    log_and_print("=" * 50)
    log_and_print(f" Best Fitness Score: {best_fitness:.2f}")
    log_and_print(f" Average Fitness Score: {avg_fitness:.2f}")
    log_and_print(f" Best Length Achieved: {best_length}")
    log_and_print(f" Average Length of Snakes: {avg_length:.2f}")
    log_and_print("-" * 50)
    log_and_print(" Inherited Weights (Brain Parameters)")
    log_and_print(f"  - Food Bonus Weight: {best_weights[0]:.3f}")
    log_and_print(f"  - Toward Food Weight: {best_weights[1]:.3f}")
    log_and_print(f"  - Away Food Penalty: {best_weights[2]:.3f}")
    log_and_print(f"  - Loop Penalty: {best_weights[3]:.3f}")
    log_and_print(f"  - Survival Bonus: {best_weights[4]:.3f}")
    log_and_print(f"  - Wall Penalty: {best_weights[5]:.3f}")
    log_and_print(f"  - Exploration Bonus: {best_weights[6]:.3f}")
    log_and_print(f"  - Momentum Bonus: {best_weights[7]:.3f}")
    log_and_print(f"  - Dead-End Penalty: {best_weights[8]:.3f}")
    log_and_print("=" * 50)

    # **Store Data for Future Analysis**
    generation_fitness.append(best_fitness)
    generation_avg_fitness.append(avg_fitness)
    generation_lengths.append(best_length)

    # **Evolve Snakes for Next Generation**
    snakes = evolve_snakes(snakes)


log_filename = "Snake_GA_Pygame/training_log.txt"


def log_and_print(*args, **kwargs):
    """ Prints output to the console and also writes it to a log file. """
    print(*args, **kwargs)  # Print to terminal
    with open(log_filename, "a") as log_file:
        print(*args, **kwargs, file=log_file)  # Also write to log file

def tournament_selection(snakes, tournament_size=CONFIG["TOURNAMENT_SIZE"]):
    participants = random.sample(snakes, tournament_size)
    return max(participants, key=lambda s: s.fitness_function())

def tournament_selection(snakes, tournament_size=CONFIG["TOURNAMENT_SIZE"]):
    """Selects a snake using tournament selection."""
    participants = random.sample(snakes, tournament_size)
    return max(participants, key=lambda s: s.fitness_function())

def evolve_snakes(snakes):
    """ Evolves the population by selecting top performers, mutating, and generating offspring """
    top_performers = sorted(snakes, key=lambda s: s.fitness_function(), reverse=True)[:10]
    new_snakes = []

    for _ in range(len(snakes)):
        parent1 = tournament_selection(top_performers)
        parent2 = tournament_selection(top_performers)
        cut = np.random.randint(0, len(parent1.brain))
        new_brain = np.concatenate((parent1.brain[:cut], parent2.brain[cut:]))

        # Adaptive mutation
        if len(generation_fitness) > 1 and generation_fitness[-1] > generation_fitness[-2]:
            mutation_factor = CONFIG["MUTATION_LOW"]
        else:
            mutation_factor = CONFIG["MUTATION_HIGH"]
        
        new_brain += np.random.randn(len(new_brain)) * mutation_factor

        new_snakes.append(SnakeAI(brain=new_brain))

        # Inject random diversity occasionally
        if random.random() < CONFIG["DIVERSITY_INJECTION_PROB"]:
            new_snakes.append(SnakeAI())

    return new_snakes[:len(snakes)]  # Ensure population size remains the same



snakes = [SnakeAI() for _ in range(50)]


def show_game_over_screen(snake, mode="manual"):
    screen.fill(BACKGROUND_GRAY)
    font_large = pygame.font.SysFont(None, 50)
    font_small = pygame.font.SysFont(None, 30)

    # **Game Over Message**
    game_over_text = font_large.render("Game Over", True, BLACK)
    screen.blit(game_over_text, (WIDTH // 2 -
                game_over_text.get_width() // 2, HEIGHT // 3))

    # **Display Final Score, Time, and Length**
    elapsed_time = round(time.time() - snake.start_time, 2)
    stats_text = [
        f"Time: {elapsed_time}s",
        f"Score: {snake.score}",
        f"Length: {snake.length}"
    ]

    stats_start_y = HEIGHT // 2 - 70  # Move stats up
    for i, text in enumerate(stats_text):
        stat_render = font_small.render(text, True, BLACK)
        screen.blit(stat_render, (WIDTH // 2 - stat_render.get_width() // 2, stats_start_y + i * 35))  # Adjust line spacing


    # **Button Positions**
    button_width, button_height = 150, 50
    button_spacing = 20
    button_y = HEIGHT // 1.4

    # **New Menu Button**
    menu_button = pygame.Rect(WIDTH // 2 - button_width * 1.5 -
                              button_spacing, button_y, button_width, button_height)
    replay_button = pygame.Rect(
        WIDTH // 2 - button_width // 2, button_y, button_width, button_height)
    quit_button = pygame.Rect(WIDTH // 2 + button_width // 2 +
                              button_spacing, button_y, button_width, button_height)

    pygame.draw.rect(screen, BROWN, menu_button, border_radius=10)
    pygame.draw.rect(screen, GREEN, replay_button, border_radius=10)
    pygame.draw.rect(screen, RED, quit_button, border_radius=10)

    # **Render Button Text**
    menu_text = font_small.render("Menu", True, WHITE)
    replay_text = font_small.render("Replay", True, WHITE)
    quit_text = font_small.render("Quit", True, WHITE)

    screen.blit(menu_text, (menu_button.x + button_width // 2 - menu_text.get_width() // 2,
                            menu_button.y + button_height // 2 - menu_text.get_height() // 2))
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
                if menu_button.collidepoint(event.pos):
                    return "menu"  # Return to Main Menu
                elif replay_button.collidepoint(event.pos):
                    return mode  # Restart same mode (manual or pretrained AI)
                elif quit_button.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()


def get_training_parameters():
    screen.fill(BACKGROUND_GRAY)
    font_large = pygame.font.SysFont(None, 50)
    font_small = pygame.font.SysFont(None, 30)
    font_table_small = pygame.font.SysFont(None, 24)

    # **Prompt User**
    title_text = font_large.render("AI Training Setup", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 4))

    # **Adjusted Input Box Positions**
    input_boxes = {
        "snakes_per_gen": pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 40, 200, 40),
        "num_generations": pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 40, 200, 40),
    }

    input_values = {"snakes_per_gen": "", "num_generations": ""}
    active_box = None

    # **Submit Button**
    submit_button = pygame.Rect(WIDTH // 2 - 50, HEIGHT // 2 + 100, 100, 40)  # Move it slightly lower

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
                if submit_button.collidepoint(event.pos):
                    if input_values["snakes_per_gen"].isdigit() and input_values["num_generations"].isdigit():
                        return int(input_values["snakes_per_gen"]), int(input_values["num_generations"])

            # Handle Keyboard Input
            if event.type == pygame.KEYDOWN:
                if active_box:
                    if event.key == pygame.K_BACKSPACE:
                        input_values[active_box] = input_values[active_box][:-1]
                    elif event.key in range(pygame.K_0, pygame.K_9 + 1):
                        input_values[active_box] += event.unicode

        # **Render Input Boxes**
        for key, box in input_boxes.items():
            pygame.draw.rect(screen, WHITE, box, border_radius=5)
            text_surface = font_small.render(input_values[key], True, BLACK)
            screen.blit(text_surface, (box.x + 10, box.y + 10))

        # **Render Labels Centered Above Input Boxes**
        label_x_offset = input_boxes["snakes_per_gen"].width // 2
        screen.blit(font_small.render("# of Snakes:", True, BLACK),
                    (input_boxes["snakes_per_gen"].x + label_x_offset - 50, input_boxes["snakes_per_gen"].y - 35))
        screen.blit(font_small.render("# of Generations:", True, BLACK),
                    (input_boxes["num_generations"].x + label_x_offset - 80, input_boxes["num_generations"].y - 35))

        # **Render Submit Button**
        pygame.draw.rect(screen, GREEN, submit_button, border_radius=10)
        submit_text = font_small.render("Start", True, WHITE)
        screen.blit(submit_text, (submit_button.x + submit_button.width // 2 - submit_text.get_width() // 2,
                                  submit_button.y + submit_button.height // 2 - submit_text.get_height() // 2))

        # **Training Recommendations Table**
        table_start_y = submit_button.y + 60  # Position the table below the Start button
        col_widths = [180, 120, 120, 220]  # Define column widths for spacing

        # Column Headers
        headers = ["Training Type", "# Snakes", "# Gens", "Best Use Case"]
        for i, header in enumerate(headers):
            header_text = font_small.render(header, True, BROWN)
            screen.blit(header_text, (WIDTH // 2 - 275 + sum(col_widths[:i]), table_start_y))

        # Training Data
        training_recommendations = [
            ("Quick Testing", "10-15", "5-10", "Test behavior changes"),
            ("Balanced Learning", "20-30", "15-25", "Good learning vs. performance"),
            ("Deep Optimization", "40-50", "30-50", "Best for AI mastery")
        ]

        # Render Table Rows
        for row, (name, snakes, gens, desc) in enumerate(training_recommendations):
            name_text = font_table_small.render(name, True, BLACK)
            snakes_text = font_table_small.render(snakes, True, BLACK)
            gens_text = font_table_small.render(gens, True, BLACK)
            desc_text = font_table_small.render(desc, True, BLACK)

            screen.blit(name_text, (WIDTH // 2 - 275, table_start_y + (row + 1) * 30))
            screen.blit(snakes_text, (WIDTH // 2 - 100, table_start_y + (row + 1) * 30))
            screen.blit(gens_text, (WIDTH // 2 + 20, table_start_y + (row + 1) * 30))
            screen.blit(desc_text, (WIDTH // 2 + 100, table_start_y + (row + 1) * 30))

        pygame.display.flip()


def show_pretrained_models():
    screen.fill(BACKGROUND_GRAY)
    font_large = pygame.font.SysFont(None, 50)
    font_small = pygame.font.SysFont(None, 30)

    title_text = font_large.render("Select Pre-Trained Model", True, BROWN)
    screen.blit(title_text, (WIDTH // 2 -
                title_text.get_width() // 2, HEIGHT // 6))

    model_buttons = []
    button_width, button_height = 200, 50
    button_y = HEIGHT // 3.2

    # Dictionary storing model explanations
    model_explanations = {
        "Hunter": "Moves aggressively toward food.",
        "Strategist": "Balances food collection with long-term survival.",
        "Explorer": "Mix of exploration and food-seeking.",
        "Risk Taker": "Adapts risky strategies for short period of time.",
        "AI Mastery": "Advanced AI with optimized weights."
    }

    hovered_button = None 

    for i, (model_name, params) in enumerate(PRETRAINED_MODELS.items()):
        button_x = WIDTH // 4 - button_width // 2
        button_rect = pygame.Rect(
            button_x, button_y + i * 80, button_width, button_height)
        model_buttons.append((button_rect, model_name, params))


    while True:
        screen.fill(BACKGROUND_GRAY)
        screen.blit(title_text, (WIDTH // 2 -
                    title_text.get_width() // 2, HEIGHT // 6))

        for button_rect, model_name, params in model_buttons:
            color = (000, 128, 000) if hovered_button == model_name else GREEN
            pygame.draw.rect(screen, color, button_rect, border_radius=10)

            text = font_small.render(model_name, True, WHITE)
            screen.blit(text, (button_rect.x + button_width // 2 - text.get_width() // 2,
                            button_rect.y + button_height // 2 - text.get_height() // 2))


            # Display model explanation next to the button
            explanation_text = font_small.render(
                model_explanations[model_name], True, BLACK)
            screen.blit(explanation_text, (button_rect.x +
                        button_width + 20, button_rect.y + 10))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEMOTION:
                hovered_button = None 
                for button_rect, model_name, params in model_buttons:
                    if button_rect.collidepoint(event.pos):
                        hovered_button = model_name  
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for button_rect, model_name, params in model_buttons:
                    if button_rect.collidepoint(event.pos):
                        return model_name, params 



def show_training_summary(best_score, best_length, training_time):
    screen.fill(BACKGROUND_GRAY)
    font_large = pygame.font.SysFont(None, 50)
    font_small = pygame.font.SysFont(None, 30)

    # Title
    title_text = font_large.render("Training Complete", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 -
                title_text.get_width() // 2, HEIGHT // 3))

    # Stats
    stats_text = [
        f"Final Best Score: {best_score_overall}",
        f"Final Best Length: {best_length_overall}",
        f"Total Training Time: {training_time:.2f}s"
    ]
    stats_start_y = HEIGHT // 2 - 40
    for i, text_line in enumerate(stats_text):
        stat_render = font_small.render(text_line, True, BLACK)
        screen.blit(stat_render, (WIDTH // 2 - stat_render.get_width() // 2,
                                  stats_start_y + i * 40))

    # --- Define Buttons in a Single Pass ---
    button_width = 120
    button_height = 40
    button_spacing = 20

    # total width for 4 buttons = 4 * button_width + 3 * button_spacing
    total_button_width = 4 * button_width + 3 * button_spacing
    start_x = (WIDTH - total_button_width) // 2
    y = int(HEIGHT * 0.75)  # 75% down the screen

    # Create rects side by side
    menu_button = pygame.Rect(start_x, y, button_width, button_height)
    pretrain_button = pygame.Rect(
        start_x + (button_width + button_spacing), y, button_width, button_height
    )
    replay_button = pygame.Rect(
        start_x + 2*(button_width +
                     button_spacing), y, button_width, button_height
    )
    quit_button = pygame.Rect(
        start_x + 3*(button_width +
                     button_spacing), y, button_width, button_height
    )

    # Draw them
    pygame.draw.rect(screen, BROWN, menu_button, border_radius=10)
    pygame.draw.rect(screen, BLUE, pretrain_button, border_radius=10)
    pygame.draw.rect(screen, GREEN, replay_button, border_radius=10)
    pygame.draw.rect(screen, RED, quit_button, border_radius=10)

    # Button text
    def draw_button_text(rect, text):
        text_surf = font_small.render(text, True, WHITE)
        screen.blit(text_surf, (rect.x + rect.width // 2 - text_surf.get_width() // 2,
                                rect.y + rect.height // 2 - text_surf.get_height() // 2))

    draw_button_text(menu_button, "Menu")
    draw_button_text(pretrain_button, "Pre-Train")
    draw_button_text(replay_button, "Replay")
    draw_button_text(quit_button, "Quit")

    pygame.display.flip()

    # --- Wait for Clicks ---
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if menu_button.collidepoint(event.pos):
                    return "menu"
                elif pretrain_button.collidepoint(event.pos):
                    return "pretrain"
                elif replay_button.collidepoint(event.pos):
                    return "replay"
                elif quit_button.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()


def run_pretrained_from_training(weights):
    print("Starting Pre-Trained AI Mode with Best Weights from Training...")
    run_pretrained_ai(weights)


def run_pretrained_ai(model_params):
    global snakes  # Ensure we update the global variable

    snake = SnakeAI(brain=model_params)
    snakes = [snake]  # Only one snake should be in the list

    running = True
    while running and snake.alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        snake.move()  # Ensure move() is executed
        draw_game(game_mode="pretrained_ai", model_params=model_params)

    # **Show Game Over Screen and Handle Replay**
    action = show_game_over_screen(snake, mode="model_selection")
    if action == "model_selection":
        # Return to Pre-Trained Model Selection
        model_name, model_params = show_pretrained_models()
        run_pretrained_ai(model_params)  # Restart with selected model
    elif action == "menu":
        selection = menu_screen()  # Return to main menu


# **Show the menu before running the game**
def main():
    while True:
        selection = menu_screen()

        if selection == "manual":
            print("Starting Manual Play...")
            run_manual_mode()

        elif selection == "train":
            best_score_overall = 0
            best_length_overall = 0
            while True:  # Allow replaying AI training
                snakes_per_gen, num_generations = get_training_parameters()

                # ✅ Reset all training history before starting new training
                generation_fitness.clear()
                generation_avg_fitness.clear()
                generation_lengths.clear()
                generation_avg_lengths.clear()

                print(
                    f"Starting AI Training with {snakes_per_gen} snakes per generation for {num_generations} generations.")

                # Initialize Snakes
                global snakes
                snakes = [SnakeAI() for _ in range(snakes_per_gen)]
                print("Population size:", len(snakes))

                training_start_time = time.time()

                for generation in range(num_generations):
                    best_score = max(
                        (snake.score for snake in snakes if snake.alive), default=0)
                    best_length = max(
                        (snake.length for snake in snakes if snake.alive), default=0)
                    elapsed_time = round(time.time() - training_start_time, 2)
                    best_snake = max(
                        snakes, key=lambda s: s.fitness_function(), default=None)
                    if best_snake:
                        # Convert numpy array to list for saving
                        best_weights = best_snake.brain.tolist()

                    log_and_print(
                        f"Generation {generation+1} - Best Score: {best_score}, Length: {best_length}, Time: {elapsed_time}s")
                    run_generation()

                # **Show Training Summary and Handle Replay**
                action = show_training_summary(best_score_overall, best_length_overall,
                                               round(time.time() - training_start_time, 2))
                if action == "replay":
                    continue
                elif action == "pretrain":
                    run_pretrained_from_training(best_weights)
                    selection = "menu"
                    break
                elif action == "menu":
                    selection = menu_screen()
                    break

        elif selection == "pretrained":
            model_name, model_params = show_pretrained_models()
            print(f"Selected {model_name} with parameters: {model_params}")
            run_pretrained_ai(model_params)  # Pass parameters to the function

        elif selection == "quit":
            print("Exiting Game...")
            pygame.quit()
            sys.exit()


if __name__ == "__main__":
    main()
