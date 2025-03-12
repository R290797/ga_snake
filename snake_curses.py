# NOTES TO DISCUSS:
# Arbirtrary values for scores, make it hard to define hyperparameters
# Bias towards directions coming first in the array, since max index that appears first ist taken

import curses
import random
import time

HEIGHT = 30
WIDTH = 60
TOP_BAR = 3  # Number of extra lines above the playing field
SNAKE_CHAR = "O"
FOOD_CHAR = "X"
EMPTY_CHAR = " "

# Directions
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

def create_food(snake):  # Ensure food does not spawn in the snake
    while True:
        food = (random.randint(1, HEIGHT - 2), random.randint(1, WIDTH - 2))
        if food not in snake.positions:
            return food

# Player Controlled Snake Class
class Snake:

    # Constructor
    def __init__(self):
        self.positions = [(HEIGHT // 2, WIDTH // 2)]
        self.direction = RIGHT
        self.food = create_food(self)
        self.score = 0

    # Move Function
    def move(self):
       
        head_y, head_x = self.positions[0]
        new_head = (head_y + self.direction[0], head_x + self.direction[1])
        
        # Check for collisions
        if (new_head in self.positions) or (new_head[0] == 0 or new_head[0] == HEIGHT - 1 or new_head[1] == 0 or new_head[1] == WIDTH - 1):
            return "collision"
        
        self.positions.insert(0, new_head)
        
        # Check if snake eats food
        if new_head == self.food:
            self.food = create_food(self)
            self.score += 1
        else:
            self.positions.pop()  # Remove tail
        
        return "safe"

# Gen AI Snake Class  
class SnakeAI:

    # Constructor
    def __init__(self, positions=[(HEIGHT // 2, WIDTH // 2)], brain=[random.randint(0, 1) for _ in range(7)]): # Default Position to Middle
        self.positions = positions # All positions of the snake - 2 Dimensional List [[Y,X]]
        self.prev_positions = [] # Track 20 last head positions (init empty)
        self.length = len(self.positions) # Length of the Current Snake
        self.moves_made = 0 # Track moves made by snake during game
        self.current_direction = None # Track current movement
        self.food = create_food(self)
        self.score = 0

        # Brain ---> List of weights for move evaluation (GENOME)
        self.brain = brain

    # STATE SPACE IDENTIFICATION FUNCTIONS (for move evaluation)

    # Check Collision
    def check_collision(self, positions):
        head_y, head_x = positions[0]  # Extract the new head position

        # Check if the position is outside the boundaries
        if head_y <= 0 or head_y >= HEIGHT - 1 or head_x <= 0 or head_x >= WIDTH - 1:
            return True  # Collision with the wall

        # Check if the position is inside the snake's body
        if (head_y, head_x) in self.positions:
            return True  # Collision with itself

    # X Distance to Food (Based on given positions)
    def food_distance_x(self, positions):
        distance_x = abs(positions[0][1] - self.food[1])
        return distance_x
    
    # Y Distance to Food (Based on given positions)
    def food_distance_y(self, positions):
        distance_y = abs(positions[0][0] - self.food[0])
        return distance_y
    
    # X Head Distance to Tail (Based on given positions)
    def tail_distance_x(self, positions):

        # Return 0 if snake is length 1
        if self.length < 2:
            return 0

        # Use Length to find tail
        distance_x = abs(positions[0][1] - positions[self.length-1][1])
        return distance_x
    
    # Y Head Distance to Tail (Based on given positions)
    def tail_distance_y(self, positions):

         # Return 0 if snake is length 1
        if self.length < 2:
            return 0

        # Use Length to find tail 
        distance_y = abs(positions[0][0] - positions[self.length-1][0])
        return distance_y
    
    # Loop Detection
    def is_looping(self, positions):

        if self.length < 5:
            return False  # Not enough history to detect a loop

        # Check if the last 10 positions are identical
        return all(pos == positions[-1] for pos in positions[-5:])
    
    # Check if position contains food
    def check_food(self, positions):
        return positions[0] == self.food

    # Evaluate Score at current position (if snake were to move in that direction)
    def eval_position(self, positions):

        # Copy current Positions (Avoid Manipulating original)
        current_snake = positions[:]

        # Use new_snake as parameter for evaluation, track score
        score = 0

        # Check Collision 
        if self.check_collision(current_snake):
            return self.brain[0] * -20 # Penalize Collision (with weight)
        
        # If no Collision, calculate other parameters (with weights)
        food_x  = self.brain[1] * -self.food_distance_x(current_snake)
        food_y = self.brain[2] * -self.food_distance_y(current_snake)
        tail_x = self.brain[3] * -self.tail_distance_x(current_snake)
        tail_y = self.brain[4] * -self.tail_distance_y(current_snake)

        # Check if Looping
        if self.is_looping(current_snake):
            score -= self.brain[5] * 10 # Penalize looping

        # Check if Food
        if self.check_food(current_snake):
            score += self.brain[6] * 100 # Reward food

        # Calculate position score
        score += food_x + food_y + tail_x + tail_y
        return score
    
    # Recursive Lookahead function for evaluation
    def eval_position_lookahead(self, direction, steps=3, decay=0.8):
        global DIRECTIONS

        # Get Next snake by moving in that direction
        head_y, head_x = self.positions[0]
        next_head = (head_y + direction[0], head_x + direction[1])
        next_snake = self.positions[:]  # Copy list by slicing, avoid manipulating original
        next_snake.insert(0, next_head)
        next_snake.pop()

        # Evaluate immediate move
        score = self.eval_position(next_snake)

        # Go through further steps
        if steps > 0:

            lookahead_scores = []

            for next_direction in DIRECTIONS:
                lookahead_score = self.eval_position_lookahead(next_direction, steps-1, decay)
                lookahead_scores.append(lookahead_score)

            # Take Highest Score
            max_score = max(lookahead_scores)

            # Apply Decay
            score += decay * max_score

        return score
    
    # TODO: ADD FITNESS FUNCTION (Metric to judge how well the snake performed)


    # Apply AI Logic in Moves
    def move(self):
        global DIRECTIONS

        # Save Scores
        direction_scores = []

        # Get Valid Direction (Snake cannot turn in on itself - e.g. up to immediate down)
        if self.current_direction == None:
            valid_directions = DIRECTIONS
        else:
            valid_directions = [d for d in DIRECTIONS if d != (-self.current_direction[0], -self.current_direction[1])]

        # Calculate Score for all Valid directions
        for direction in valid_directions: 
            direction_score = self.eval_position_lookahead(direction, steps=5)
            direction_scores.append(direction_score)

        # Get Index of Max score
        max_score = max(direction_scores)

        # Random Tie breaker to counteract indecisiveness
        max_indices = [i for i, score in enumerate(direction_scores) if score == max_score]
        chosen_index = random.choice(max_indices)
        chosen_direction = valid_directions[chosen_index]

        # Set Current Direction
        self.current_direction = valid_directions[chosen_index]

        # Render Move and Game Logic
        head_y, head_x = self.positions[0]
        new_head = (head_y + chosen_direction[0], head_x + chosen_direction[1])
        
        # Check for collisions
        if (new_head in self.positions) or (new_head[0] == 0 or new_head[0] == HEIGHT - 1 or new_head[1] == 0 or new_head[1] == WIDTH - 1):
            return "collision"
        
        # Snake had not Collided, Append move to previous moves
        self.prev_positions.append(new_head)

        # Ensure no more than 5 moves are being saved
        if len(self.prev_positions) > 20:
            self.prev_positions.pop(0)

        self.positions.insert(0, new_head)

        # Count Move
        self.moves_made += 1
    
        # Check if snake eats food
        if new_head == self.food:
            self.food = create_food(self)
            self.score += 1
        else:
            self.positions.pop()  # Remove tail
        
        return "safe"


# Display the Board in the Terminal via Curses
def draw_board(stdscr, snake: Snake):
    stdscr.clear()  # Clear screen before redrawing
    
    # Draw top bar with game info
    stdscr.addstr(0, 0, f"Score: {snake.score}")
    stdscr.addstr(1, 0, "Press 'q' to quit")
    stdscr.addstr(2, 0, "=========================")
    
    # Draw top border
    stdscr.addstr(TOP_BAR, 0, "#" * WIDTH)

    for y in range(1, HEIGHT - 1):
        row = ["#"]  # Left border
        for x in range(1, WIDTH - 1):
            if (y, x) in snake.positions:
                row.append(SNAKE_CHAR)
            elif (y, x) == snake.food:
                row.append(FOOD_CHAR)
            else:
                row.append(EMPTY_CHAR)
        row.append("#")  # Right border
        stdscr.addstr(y + TOP_BAR, 0, "".join(row))  # Print row at correct offset

    # Draw bottom border
    stdscr.addstr(HEIGHT - 1 + TOP_BAR, 0, "#" * WIDTH)

    stdscr.refresh()  # Refresh screen



# Display the Board in the Terminal via Curses (For AI)
def draw_board_AI(stdscr, snake: SnakeAI):
    stdscr.clear()  # Clear screen before redrawing
    
    # Draw top bar with game info
    stdscr.addstr(0, 0, f"X pos: {snake.food_distance_x(snake.positions)}, Y pos: {snake.food_distance_y(snake.positions)}")
    stdscr.addstr(1, 0, f"Moved Made: {snake.moves_made}, previous moves: {snake.prev_positions}")
    stdscr.addstr(2, 0, "=========================")
    
    # Draw top border
    stdscr.addstr(TOP_BAR, 0, "#" * WIDTH)

    # Draw Game Area between left and right border
    for y in range(1, HEIGHT - 1):
        row = ["#"]  # Left border
        for x in range(1, WIDTH - 1):
            if (y, x) in snake.positions:
                row.append(SNAKE_CHAR)
            elif (y, x) == snake.food:
                row.append(FOOD_CHAR)
            else:
                row.append(EMPTY_CHAR)
        row.append("#")  # Right border
        stdscr.addstr(y + TOP_BAR, 0, "".join(row))  # Print row at correct offset

    # Draw bottom border
    stdscr.addstr(HEIGHT - 1 + TOP_BAR, 0, "#" * WIDTH)

    stdscr.refresh()  # Refresh screen

def run_game(stdscr):

    # Track Game State
    active = True
    game_state = "Training"
    prev_game_state = "Training" # Save Previous Game State that was not "Menu"

    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(1)  # Non-blocking input
    stdscr.timeout(100)  # Refresh rate (controls speed)

    snake = Snake()
    snakeAI = SnakeAI()

    # Game Loop
    while active:

        # Normal Snake (Playing the Game Manually) - Catch Key Strokes
        if game_state == "Running":
            key = stdscr.getch()
            if key == ord("w") and snake.direction != DOWN:
                snake.direction = UP
            elif key == ord("s") and snake.direction != UP:
                snake.direction = DOWN
            elif key == ord("a") and snake.direction != RIGHT:
                snake.direction = LEFT
            elif key == ord("d") and snake.direction != LEFT:
                snake.direction = RIGHT
            elif key == ord("q"):
                game_state = "Game Over"

            snake_status = snake.move()
            draw_board(stdscr, snake)
            time.sleep(0.01)  # Game speed

            if snake_status == "collision":
                game_state = "Game Over"



        # Training Loop
        if game_state == "Training":

            # Apply AI Direction Evaluation for move
            snakeAI.move()

            snake_status = snakeAI.move()
            draw_board_AI(stdscr, snakeAI)
            time.sleep(0.01)  # Game speed

            if snake_status == "collision":
                game_state = "Game Over"


        # Playground Loop

    
        if game_state == "Game Over":

            # Game Over Screen
            stdscr.clear()
            stdscr.addstr(1, 0, "Game Over" )
            stdscr.addstr(2, 0, 'Press "r" to try again')
            stdscr.addstr(3, 0, 'Press "q" to quit')

            # Refresh Screen
            stdscr.refresh()

            # Catch Key Strokes
            key = stdscr.getch()
            if key == ord("r"):
                snakeAI = SnakeAI()
                game_state = prev_game_state
                continue

            elif key == ord("q"):
                break

            

curses.wrapper(run_game)
