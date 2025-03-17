import pygame
import random
import sys
import time

# Initialize Pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 600, 600
TOP_BAR_HEIGHT = 50
SIDE_BAR_WIDTH = 200
PADDING = 20
CELL_SIZE = 20
GAP = 2
FPS = 10

# Colors (Same as AI-trained version)
WHITE = (255, 255, 255)
BACKGROUND_GRAY = (215, 210, 203)
GRAY = (100, 100, 100)
GREEN = (21, 71, 52)  # Snake color
RED = (128, 5, 0)  # Food color
BLACK = (0, 0, 0)  # Background
BROWN = (139, 69, 19)  # Walls

# Directions (UP, DOWN, LEFT, RIGHT)
DIRECTIONS = {
    pygame.K_UP: (0, -1),
    pygame.K_DOWN: (0, 1),
    pygame.K_LEFT: (-1, 0),
    pygame.K_RIGHT: (1, 0),
}

# Setup Game Window (Same size and layout as AI mode)
screen = pygame.display.set_mode((WIDTH, HEIGHT + TOP_BAR_HEIGHT))
pygame.display.set_caption("Snake - Manual Play (Arrow Keys)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

# Adjust game area with padding
GAME_AREA_X = PADDING
GAME_AREA_Y = PADDING + TOP_BAR_HEIGHT
GAME_AREA_WIDTH = WIDTH - 2 * PADDING
GAME_AREA_HEIGHT = HEIGHT - 2 * PADDING


class ManualKeysSnake:
    def __init__(self):
        # ✅ Initialize Snake First
        self.snake = [(GAME_AREA_X + GAME_AREA_WIDTH // 2,
                       GAME_AREA_Y + GAME_AREA_HEIGHT // 2)]
        self.direction = DIRECTIONS[pygame.K_RIGHT]
        self.score = 0
        self.length = 0
        self.alive = True
        self.start_time = time.time()
        self.last_food_time = time.time()
        self.moves_made = 0  # ✅ Track total moves
        self.food_collected = 0  # ✅ Track total food eaten


        # ✅ Define Border Walls Before Spawning Food
        self.border_walls = set()
        for x in range(GAME_AREA_X - CELL_SIZE, GAME_AREA_X + GAME_AREA_WIDTH + CELL_SIZE, CELL_SIZE):
            self.border_walls.add((x, GAME_AREA_Y - CELL_SIZE))  # Top border
            self.border_walls.add(
                (x, GAME_AREA_Y + GAME_AREA_HEIGHT))  # Bottom border

        for y in range(GAME_AREA_Y - CELL_SIZE, GAME_AREA_Y + GAME_AREA_HEIGHT + CELL_SIZE, CELL_SIZE):
            self.border_walls.add((GAME_AREA_X - CELL_SIZE, y))  # Left border
            self.border_walls.add(
                (GAME_AREA_X + GAME_AREA_WIDTH, y))  # Right border

        # ✅ Now Spawn Food After Everything is Initialized
        self.food = self.spawn_food()

    def spawn_food(self):
        while True:
            food_x = random.randrange(
                GAME_AREA_X, GAME_AREA_X + GAME_AREA_WIDTH, CELL_SIZE)
            food_y = random.randrange(
                GAME_AREA_Y, GAME_AREA_Y + GAME_AREA_HEIGHT, CELL_SIZE)
            if (food_x, food_y) not in self.snake and (food_x, food_y) not in self.border_walls:
                return food_x, food_y

    def move(self):
        head_x, head_y = self.snake[0]
        new_head = (
            head_x + self.direction[0] * CELL_SIZE, head_y + self.direction[1] * CELL_SIZE)

        # **Fix Wall Collision Detection**
        if (
            new_head in self.snake  # Self-collision
            or new_head[0] < GAME_AREA_X  # Hits left wall
            or new_head[0] >= GAME_AREA_X + GAME_AREA_WIDTH  # Hits right wall
            or new_head[1] < GAME_AREA_Y  # Hits top wall
            # Hits bottom wall
            or new_head[1] >= GAME_AREA_Y + GAME_AREA_HEIGHT
        ):
            self.alive = False
            return  # ✅ Prevents further execution

        # Move the snake
        self.snake.insert(0, new_head)
        self.moves_made += 1
        
        # **Fix Food Collection Detection**
        if new_head == self.food:  # ✅ Checks correct position
            self.score += 50
            self.length += 1
            self.food = self.spawn_food()
            self.last_food_time = time.time()
        else:
            self.snake.pop()  # Move the snake

        # **Fix Starvation Mechanism**
        if time.time() - self.last_food_time > 10:
            self.alive = False  # ✅ Only starve if 10s passes without food


def draw_game(snake):
    screen.fill(BACKGROUND_GRAY)
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH +
                     SIDE_BAR_WIDTH, TOP_BAR_HEIGHT))

    # **Display Score, Length, and Time**
    elapsed_time = round(time.time() - snake.start_time, 2)
    score_text = font.render(f"Score: {snake.score}", True, WHITE)
    time_text = font.render(f"Time: {elapsed_time}s", True, WHITE)
    length_text = font.render(f"Length: {snake.length}", True, WHITE)

    # **Position Each Text in the Top Bar**
    section_width = WIDTH // 3
    screen.blit(score_text, (section_width * 0.2, 10))
    screen.blit(time_text, (section_width * 1.2, 10))
    screen.blit(length_text, (section_width * 2.2, 10))

    # ✅ **Adjust Food Border Radius to Match AI Mode**
    pygame.draw.rect(
        screen, RED, (snake.food[0] + 2, snake.food[1] + 2, CELL_SIZE - 4, CELL_SIZE - 4), border_radius=5)

    # ✅ **Adjust Snake Border Radius to Match AI Mode**
    for segment in snake.snake:
        pygame.draw.rect(
            screen, GREEN, (segment[0] + 2, segment[1] + 2, CELL_SIZE - 4, CELL_SIZE - 4), border_radius=5)

    # ✅ **Adjust Walls Border Radius to Match AI Mode**
    for wall in snake.border_walls:
        pygame.draw.rect(
            screen, BROWN, (wall[0], wall[1], CELL_SIZE, CELL_SIZE), border_radius=5)

    pygame.display.flip()


def show_game_over_screen(snake):
    """Display a game over screen with game results and wait for user input to either replay or return to menu."""
    screen.fill(BACKGROUND_GRAY)
    font_large = pygame.font.SysFont(None, 50)
    font_small = pygame.font.SysFont(None, 30)

    # **Game Over Message**
    game_over_text = font_large.render("Game Over", True, RED)
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2 - 150))

    # **Calculate Game Stats**
    elapsed_time = round(time.time() - snake.start_time, 2)
    stats_text = [
        f"Time: {elapsed_time}s",
        f"Score: {snake.score}",
        f"Length: {snake.length}",
    ]

    # **Display Game Stats**
    stats_start_y = HEIGHT // 2 - 80  # Move stats higher to avoid overlapping buttons
    for i, text in enumerate(stats_text):
        stat_render = font_small.render(text, True, BLACK)
        screen.blit(stat_render, (WIDTH // 2 - stat_render.get_width() // 2, stats_start_y + i * 30))

    # **Button Positions**
    button_width, button_height = 150, 50
    button_spacing = 20

    menu_button = pygame.Rect(WIDTH // 2 - button_width - button_spacing // 2, HEIGHT // 2 + 50, button_width, button_height)
    replay_button = pygame.Rect(WIDTH // 2 + button_spacing // 2, HEIGHT // 2 + 50, button_width, button_height)

    pygame.draw.rect(screen, BROWN, menu_button, border_radius=10)
    pygame.draw.rect(screen, GREEN, replay_button, border_radius=10)

    # **Render Button Text**
    menu_text = font_small.render("Menu", True, WHITE)
    replay_text = font_small.render("Replay", True, WHITE)
    screen.blit(menu_text, (menu_button.x + menu_button.width // 2 - menu_text.get_width() // 2,
                            menu_button.y + menu_button.height // 2 - menu_text.get_height() // 2))
    screen.blit(replay_text, (replay_button.x + replay_button.width // 2 - replay_text.get_width() // 2,
                              replay_button.y + replay_button.height // 2 - replay_text.get_height() // 2))

    pygame.display.flip()

    # **Wait for User Input**
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if menu_button.collidepoint(pos):
                    return "menu"
                elif replay_button.collidepoint(pos):
                    return "replay"
    return "menu"


def run_manual_mode():
    snake = ManualKeysSnake()

    while snake.alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Handle Key Press for Direction Change
            if event.type == pygame.KEYDOWN:
                if event.key in DIRECTIONS:
                    new_direction = DIRECTIONS[event.key]
                    # Prevent reversing
                    if new_direction != (-snake.direction[0], -snake.direction[1]):
                        snake.direction = new_direction

        snake.move()
        draw_game(snake)
        clock.tick(FPS)
    action = show_game_over_screen(snake)
    return action
