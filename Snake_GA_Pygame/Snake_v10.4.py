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

PRETRAINED_MODELS = {
    "Hunter": [1.35, 0.90, 0.40, 0.02, -1.00, -1.50, -1.10, 4.20, -1.30],
    "Strategist": [1.10, 0.80, 0.30, 0.05, 1.50, 0.50, -1.30, 3.70, -1.60],
    "Explorer": [1.20, 0.85, 0.50, 0.03, -0.80, -0.80, -0.90, 4.00, -1.50],
    "Risk Taker": [1.50, 0.95, 0.35, 0.01, -2.00, -2.00, -1.40, 4.50, -1.20]
}

# AI-Controlled Snake Class


class SnakeAI:
    def __init__(self, brain=None):
        self.snake = [(GAME_AREA_X + GAME_AREA_WIDTH // 2,
                       GAME_AREA_Y + GAME_AREA_HEIGHT // 2)]
        self.direction = random.choice(DIRECTIONS)
        self.moves_made = 0  # Track the total moves the snake makes
        self.score = 0  # Score of the snake
        self.fitness_score = 0  # Fitness score of the snake
        self.length = 0    # Length of the snake
        self.alive = True
        self.start_time = time.time()
        self.last_food_time = time.time()  # Time when the last food was collected

        # AI Weights
        if brain is None:
            # Smaller range for stability
            self.brain = np.random.uniform(-0.75, 0.75, 9)
            self.brain /= np.linalg.norm(self.brain)  # Keep values balanced
        else:
            self.brain = np.array(brain)  # Ensure it's a NumPy array
        self.previous_positions = []

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

    def detect_loop(self):
        """ Detects if the snake is stuck in a repetitive loop and dynamically adjusts penalties. """
        loop_length = max(6, min(12, self.length // 2)
                          )  # Dynamic loop length based on snake size
        # Adjust based on snake length
        history_window = max(15, self.length * 2)

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
        fitness = (self.score * self.length * 5) + (time.time() -
                                                    self.start_time) * 10 - (self.moves_made / (self.score + 1))

        food_streak_bonus = (self.score ** 1.5) * 2  # Encourages streaks
        fitness += food_streak_bonus

        return fitness

    def choose_direction(self):
        def simulate_move(head, direction, depth=0):
            """Simulates a move and returns a heuristic score based on the resulting state."""
            new_x, new_y = head[0] + direction[0] * \
                CELL_SIZE, head[1] + direction[1] * CELL_SIZE

            # Collision or wall check
            if (new_x, new_y) in self.snake or (new_x, new_y) in self.border_walls:
                return -1000  # Heavy penalty for moving into a wall or itself

            # Calculate distance to food
            distance_to_food = abs(
                self.food[0] - new_x) + abs(self.food[1] - new_y)

            #  Reduce food bonus (75 instead of 100) to prevent overriding safety concerns**
            food_bonus = self.brain[0] * \
                (100 if (new_x, new_y) == self.food else 0)

            #  Remove conflicting away-food penalty**
            toward_food_reward = self.brain[1] * -distance_to_food

            #  Make loop penalty progressive (-10 per visit, up to -30 max)**
            visit_count = self.previous_positions.count((new_x, new_y))
            loop_penalty = self.brain[3] * \
                (-20 * visit_count if visit_count > 1 else 0)

            #  Increase wall penalty (-5 instead of -2) for better obstacle avoidance**
            is_near_wall = new_x < CELL_SIZE or new_x > WIDTH - \
                CELL_SIZE or new_y < CELL_SIZE or new_y > HEIGHT - CELL_SIZE
            is_food_near_wall = self.food[0] < CELL_SIZE or self.food[0] > WIDTH - \
                CELL_SIZE or self.food[1] < CELL_SIZE or self.food[1] > HEIGHT - CELL_SIZE
            wall_penalty = self.brain[5] * \
                (-3 if is_near_wall and not is_food_near_wall else 0)

            # Reduce momentum bonus (5 instead of 10) & make it progressive
            # Check last 5 moves
            recent_direction = self.previous_positions[-10:]
            same_direction_count = sum(
                1 for pos in recent_direction if pos == self.direction)
            momentum_bonus = self.brain[7] * \
                (10 if same_direction_count >= 3 else 0)

            #   Apply exponential decay to exploration bonus (more important early-game)
            unique_positions = len(set(self.previous_positions))
            exploration_bonus = self.brain[6] * (unique_positions / (
                len(self.previous_positions) + 1)) * np.exp(-0.05 * len(self.previous_positions))

            #   Merge dead-end detection with self-collision penalty
            lookahead_positions = [
                (new_x + dx * CELL_SIZE, new_y + dy * CELL_SIZE) for dx, dy in DIRECTIONS]
            lookahead_collisions = sum(
                pos in self.snake for pos in lookahead_positions)
            dead_end_penalty = self.brain[8] * \
                (-20 if lookahead_collisions >= 2 else 0)

            #  Improve recursive lookahead to break loops**
            if depth > 0 and visit_count > 1:
                loop_penalty -= 10 * depth  # Increase penalty deeper in recursion

            # Compute total score
            total_score = (
                food_bonus + toward_food_reward +
                loop_penalty + wall_penalty +
                exploration_bonus + momentum_bonus + dead_end_penalty
            )

            # *Recursive Lookahead (Modified)
            if depth < 2:  # Look ahead up to 3 moves
                future_scores = [simulate_move(
                    (new_x, new_y), next_move, depth + 1) for next_move in DIRECTIONS]
                best_future_score = max(future_scores)

                #  If future best move still results in a loop, force a different choice
                if best_future_score < -50:
                    total_score -= 30  # Stronger loop deterrent
                else:
                    total_score += best_future_score * 0.7  # Discounted future rewards

            return total_score

        # Evaluate All Possible First Moves
        head_x, head_y = self.snake[0]
        best_move = None
        best_score = float('-inf')

        for move in DIRECTIONS:
            score = simulate_move((head_x, head_y), move)
            if score > best_score:
                best_score = score
                best_move = move

        # Fallback to random if no move is found
        return best_move if best_move else random.choice(DIRECTIONS)

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


def evolve_snakes(snakes):
    """ Evolves the population by selecting top performers, mutating, and generating offspring """
    top_performers = sorted(
        snakes, key=lambda s: s.fitness_function(), reverse=True)[:10]
    winner_snake = top_performers[0]
    new_snakes = []

    for _ in range(len(snakes)):
        parent1 = random.choice(top_performers[:3])  # Top 3 performers
        parent2 = random.choice(top_performers[3:])  # Mid-tier performers
        parent3 = random.choice(snakes)
        parent4 = winner_snake

        # Crossover: Mix weights from two parents
        cut = np.random.randint(0, len(parent1.brain))  # Random split point
        new_brain = np.concatenate((parent1.brain[:cut], parent2.brain[cut:]))

        # Mutation
        if len(generation_fitness) > 1 and generation_fitness[-1] > generation_fitness[-2]:
            # Less mutation if improving
            mutation = np.random.randn(len(new_brain)) * 0.1
        else:
            # More mutation if not improving
            mutation = np.random.randn(len(new_brain)) * 0.3

        new_brain += mutation

        # **Create new snake with evolved brain**
        new_snakes.append(SnakeAI(brain=new_brain))

    return new_snakes


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

    for i, text in enumerate(stats_text):
        stat_render = font_small.render(text, True, BLACK)
        screen.blit(stat_render, (WIDTH // 2 -
                    stat_render.get_width() // 2, HEIGHT // 2 + i * 40))

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

    # **Prompt User**
    title_text = font_large.render("AI Training Setup", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 -
                title_text.get_width() // 2, HEIGHT // 4))

# **Adjusted Input Box Positions**
    input_boxes = {
        # Move down slightly
        "snakes_per_gen": pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 40, 200, 40),
        # More spacing from first box
        "num_generations": pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 40, 200, 40),
    }

    input_values = {"snakes_per_gen": "", "num_generations": ""}
    active_box = None

    # **Submit Button**
    submit_button = pygame.Rect(WIDTH // 2 - 50, HEIGHT // 2 + 80, 100, 40)

    while True:
        screen.fill(BACKGROUND_GRAY)
        screen.blit(title_text, (WIDTH // 2 -
                    title_text.get_width() // 2, HEIGHT // 4))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Handle Clicks on Input Boxes
            if event.type == pygame.MOUSEBUTTONDOWN:
                for key, box in input_boxes.items():
                    if box.collidepoint(event.pos):
                        active_box = key
                # If submit button clicked
                if submit_button.collidepoint(event.pos):
                    if input_values["snakes_per_gen"].isdigit() and input_values["num_generations"].isdigit():
                        return int(input_values["snakes_per_gen"]), int(input_values["num_generations"])

            # Handle Keyboard Input
            if event.type == pygame.KEYDOWN:
                if active_box:
                    if event.key == pygame.K_BACKSPACE:
                        input_values[active_box] = input_values[active_box][:-1]
                    # Only allow numbers
                    elif event.key in range(pygame.K_0, pygame.K_9 + 1):
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
        "Risk Taker": "Adapts risky strategies for short period of time."
    }

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
            pygame.draw.rect(screen, GREEN, button_rect, border_radius=10)
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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for button_rect, model_name, params in model_buttons:
                    if button_rect.collidepoint(event.pos):
                        return model_name, params  # Return selected model


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
        f"Final Best Score: {best_score}",
        f"Final Best Length: {best_length}",
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
                        f"Generation {generation} - Best Score: {best_score}, Length: {best_length}, Time: {elapsed_time}s")
                    run_generation()

                # **Show Training Summary and Handle Replay**
                action = show_training_summary(best_score, best_length, round(
                    time.time() - training_start_time, 2))
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
