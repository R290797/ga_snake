import pygame
import random

# SNAKE GENETIC ALGORITHM PLAYGROUND

# Init Pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 600, 600  # Must be divisible by 20
TOP_BAR_HEIGHT = 50
SIDE_BAR_WIDTH = 200
PADDING = 20  # Variable padding for the game grid

CELL_SIZE = 20
GAP = 2

# Colors
WHITE = (255, 255, 255)
BACKGROUND_GRAY = (70, 70, 70)
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Screen Configurations
screen = pygame.display.set_mode((WIDTH + SIDE_BAR_WIDTH, HEIGHT + TOP_BAR_HEIGHT))
pygame.display.set_caption("Snake GA Playground")

# Font setup
font_path = "Assets/PressStart2P-Regular.ttf"
font = pygame.font.Font(font_path, 15)

# Adjust game area with padding
GAME_AREA_X = PADDING
GAME_AREA_Y = PADDING + TOP_BAR_HEIGHT
GAME_AREA_WIDTH = WIDTH - 2 * PADDING
GAME_AREA_HEIGHT = HEIGHT - 2 * PADDING

# Snake setup
snake = [(GAME_AREA_X + GAME_AREA_WIDTH // 2, GAME_AREA_Y + GAME_AREA_HEIGHT // 2)]
direction = RIGHT
food = (random.randrange(GAME_AREA_X, GAME_AREA_X + GAME_AREA_WIDTH, CELL_SIZE),
        random.randrange(GAME_AREA_Y, GAME_AREA_Y + GAME_AREA_HEIGHT, CELL_SIZE))

clock = pygame.time.Clock()
score = 0

# Game State Variables
running = True
game_state = "Menu"
game_over = False

# Display Text on Screen function
def display_text(text, x, y):
    global screen

    display_text = font.render(text, True, WHITE)
    screen.blit(display_text, (x, y))


# Event Handler
def handle_events(events):
    global running, direction, game_state

    for event in events:
        if event.type == pygame.QUIT:
            running = False

        elif game_state == "Menu" and event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            for text, x, y in buttons:
                if x <= mouse_x <= x + 200 and y <= mouse_y <= y + 40:
                    if text == "Train":
                        game_state = "Train"
                    elif text == "Use Pretrained":
                        game_state = "Pretrained"
                    elif text == "Playground":
                        game_state = "Playground"
                    elif text == "Quit":
                        running = False

        elif game_state in ["Train", "Playground"]:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction != DOWN:
                    direction = UP
                elif event.key == pygame.K_DOWN and direction != UP:
                    direction = DOWN
                elif event.key == pygame.K_LEFT and direction != RIGHT:
                    direction = LEFT
                elif event.key == pygame.K_RIGHT and direction != LEFT:
                    direction = RIGHT

# Game Loop
while running:
    screen.fill(BACKGROUND_GRAY)

    # Game State Check
    if game_state == "Menu":
        # Button setup
        button_font = pygame.font.Font(font_path, 20)
        buttons = [
            ("Train", WIDTH // 2 - 100, HEIGHT // 2 - 80),
            ("Use Pretrained", WIDTH // 2 - 100, HEIGHT // 2 - 20),
            ("Playground", WIDTH // 2 - 100, HEIGHT // 2 + 40),
            ("Quit", WIDTH // 2 - 100, HEIGHT // 2 + 100)
        ]

        for text, x, y in buttons:
            pygame.draw.rect(screen, BLACK, (x, y, 350, 40), border_radius=10)
            label = button_font.render(text, True, WHITE)
            screen.blit(label, (x + 20, y + 10))

        pygame.display.flip()
        handle_events(pygame.event.get())

    elif game_state == "Train":

        # Draw top bar
        pygame.draw.rect(screen, BACKGROUND_GRAY, (0, 0, WIDTH + SIDE_BAR_WIDTH, TOP_BAR_HEIGHT))
        header_text = font.render(f"Training", True, WHITE)
        screen.blit(header_text, (20, 10))
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (20, 40))

        # Draw Side bar
        header_text = font.render(f"Training", True, WHITE)
        screen.blit(header_text, (20, 10))

        # Draw game grid with padding
        for x in range(GAME_AREA_X, GAME_AREA_X + GAME_AREA_WIDTH, CELL_SIZE):
            for y in range(GAME_AREA_Y, GAME_AREA_Y + GAME_AREA_HEIGHT, CELL_SIZE):
                pygame.draw.rect(screen, GRAY, (x + GAP, y + GAP, CELL_SIZE - GAP * 2, CELL_SIZE - GAP * 2), border_radius=5)

        # Event Handling
        handle_events(pygame.event.get())

        # Move Snake
        head_x, head_y = snake[0]
        new_head = (head_x + direction[0] * CELL_SIZE, head_y + direction[1] * CELL_SIZE)

        # Collision Detection (Wall or Self)
        if new_head in snake or not (GAME_AREA_X <= new_head[0] < GAME_AREA_X + GAME_AREA_WIDTH and
                                     GAME_AREA_Y <= new_head[1] < GAME_AREA_Y + GAME_AREA_HEIGHT):
            running = False

        snake.insert(0, new_head)

        # Check Food Collision
        if new_head == food:
            score += 1
            food = (random.randrange(GAME_AREA_X, GAME_AREA_X + GAME_AREA_WIDTH, CELL_SIZE),
                    random.randrange(GAME_AREA_Y, GAME_AREA_Y + GAME_AREA_HEIGHT, CELL_SIZE))
        else:
            snake.pop()

        # Draw Snake
        for segment in snake:
            pygame.draw.rect(screen, GREEN, (segment[0] + GAP, segment[1] + GAP, CELL_SIZE - GAP * 2, CELL_SIZE - GAP * 2), border_radius=5)

        # Draw Food
        pygame.draw.rect(screen, RED, (food[0] + GAP, food[1] + GAP, CELL_SIZE - GAP * 2, CELL_SIZE - GAP * 2), border_radius=5)

        pygame.display.flip()
        clock.tick(10)  # Game Speed

pygame.quit()
