# NOTES TO DISCUSS:
# Arbirtrary values for scores, make it hard to define hyperparameters
# Bias towards directions coming first in the array, since max index that appears first ist taken
# curses-windows for windows

import curses
import random
import time

HEIGHT = 30
WIDTH = 60
TOP_BAR = 6  # Number of extra lines above the playing field
SNAKE_CHAR = "O"
FOOD_CHAR = "X"
EMPTY_CHAR = " "

# Directions
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]
GAME_SPEED = 0.001
MENU_OPEN = False

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
        self.length = 1
        self.moves_made = 0

    def fitness_function(self):

        # Score to judge the performance of the candidate
        # Take into consideration Food Eaten and Time Survived, but also time taken for each score on average
        # Reward High Scores and longer time survived
        # Penalize excessive moves per score
        # Score squared - average moves per score

        fitness = (self.score*self.score) - (self.moves_made/(self.score + 1))
        return fitness

    # Move Function
    def move(self):
       
        head_y, head_x = self.positions[0]
        new_head = (head_y + self.direction[0], head_x + self.direction[1])
        
        # Check for collisions
        if (new_head in self.positions) or (new_head[0] == 0 or new_head[0] == HEIGHT - 1 or new_head[1] == 0 or new_head[1] == WIDTH - 1):
            return "collision"
        
        self.positions.insert(0, new_head)

        # No Collision have Occured
        self.moves_made += 1
        self.length += 1
        
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
    def __init__(self, positions=None, brain=None): # Default Position to Middle
        self.positions = positions if positions else [(HEIGHT // 2, WIDTH // 2)] # All positions of the snake - 2 Dimensional List [[Y,X]]
        self.prev_positions = [] # Track 20 last head positions (init empty)
        self.length = len(self.positions) # Length of the Current Snake
        self.moves_made = 0 # Track moves made by snake during game
        self.current_direction = None # Track current movement
        self.food = create_food(self)
        self.score = 0
        self.fitness_score = 0

        # Brain ---> List of weights for move evaluation (GENOME)
        self.brain = brain if brain else [random.uniform(0, 1) for _ in range(7)]


    # STATE SPACE IDENTIFICATION FUNCTIONS (for move evaluation)
    #___________________________________________________________

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
    
    # Fitness Function Score
    def fitness_function(self):

        # Score to judge the performance of the candidate
        # Take into consideration Food Eaten and Time Survived, but also time taken for each score on average
        # Reward High Scores and longer time survived
        # Penalize excessive moves per score
        # Score squared - average moves per score

        fitness = (self.score*self.score) - (self.moves_made/(self.score + 1))
        return fitness


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
            direction_score = self.eval_position_lookahead(direction, steps=6) # 5/6 Steps for best Application Performance (7 Steps and above is CPU intensive)
            # Consider: For 7 Steps 3^7 Possible positions must be scored...

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

        # Ensure Prev moves is not too long
        if len(self.prev_positions) > 50:
            self.prev_positions.pop(0)

        self.positions.insert(0, new_head)

        # Count Move
        self.moves_made += 1

        # Calculate Fitness
        self.fitness_score = self.fitness_function()
    
        # Check if snake eats food
        if new_head == self.food:
            self.food = create_food(self)
            self.score += 1
            self.length += 1 # Increment Length
        else:
            self.positions.pop()  # Remove tail
        
        return "safe"


# Display the Board in the Terminal via Curses
def draw_board(stdscr, snake: Snake):
    stdscr.clear()  # Clear screen before redrawing
    
    # Draw top bar with game info
    stdscr.addstr(0, 0, "Human Snake Control", curses.A_BOLD)
    stdscr.addstr(2, 0, "Control with 'WASD', Press 'q' to quit")
    stdscr.addstr(3, 0, f"Score: {snake.score}")
    stdscr.addstr(4, 0, f"Moves Made: {snake.moves_made}") 
    stdscr.addstr(5, 0, f"Fitness Score: {snake.fitness_function()}")
    
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
    global GAME_SPEED
    stdscr.clear()  # Clear screen before redrawing
    
    # Draw top bar with game info
    stdscr.addstr(0, 0, f"Score: {snake.score}")
    stdscr.addstr(1, 0, f"Moves Made: {snake.moves_made}")
    stdscr.addstr(2, 0, f"Length: {snake.length}")
    stdscr.addstr(4, 0, f"Increase and decrease game speed with 'w' and 's'")
    stdscr.addstr(5, 0, f"Current Game Speed: {GAME_SPEED}")
    stdscr.addstr(6, 0, "    ")
    
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

        # Write extra text to the right of the game area
        info_offset = WIDTH + 2  # Set offset (2 spaces after the border)
        if y == 1:  # Display text at a specific row (adjust as needed)
            stdscr.addstr(y + TOP_BAR, info_offset, f"Candidate 1")
        elif y == 2:
            stdscr.addstr(y + TOP_BAR, info_offset, f"Collision Weight: {snake.brain[0]}")
        elif y == 3:
            stdscr.addstr(y + TOP_BAR, info_offset, f"Food X Weight: {snake.brain[1]}")
        elif y == 4:
            stdscr.addstr(y + TOP_BAR, info_offset, f"Food Y Weight: {snake.brain[2]}")
        elif y == 5:
            stdscr.addstr(y + TOP_BAR, info_offset, f"Tail X Weight: {snake.brain[3]}")
        elif y == 6:
            stdscr.addstr(y + TOP_BAR, info_offset, f"Tail Y Weight: {snake.brain[4]}")
        elif y == 7:
            stdscr.addstr(y + TOP_BAR, info_offset, f"Loop Weight: {snake.brain[5]}")
        elif y == 8:
            stdscr.addstr(y + TOP_BAR, info_offset, f"Food Collection Weight: {snake.brain[6]}")
        elif y == 11:
            stdscr.addstr(y + TOP_BAR, info_offset, f"Fitness Score: {snake.fitness_score}")
        
          

    # Draw bottom border
    stdscr.addstr(HEIGHT - 1 + TOP_BAR, 0, "#" * WIDTH)

    stdscr.refresh()  # Refresh screen


# Snake Title Card
def get_snake_menu_text():
  
    title = r"""
  ██████  ███▄    █  ▄▄▄       ██ ▄█▀▓█████      ▄████  ▄▄▄       
▒██    ▒  ██ ▀█   █ ▒████▄     ██▄█▒ ▓█   ▀     ██▒ ▀█▒▒████▄    
░ ▓██▄   ▓██  ▀█ ██▒▒██  ▀█▄  ▓███▄░ ▒███      ▒██░▄▄▄░▒██  ▀█▄  
  ▒   ██▒▓██▒  ▐▌██▒░██▄▄▄▄██ ▓██ █▄ ▒▓█  ▄    ░▓█  ██▓░██▄▄▄▄██ 
▒██████▒▒▒██░   ▓██░ ▓█   ▓██▒▒██▒ █▄░▒████▒   ░▒▓███▀▒ ▓█   ▓██▒░
▒ ▒▓▒ ▒ ░░ ▒░   ▒ ▒  ▒▒   ▓▒█░▒ ▒▒ ▓▒░░ ▒░ ░    ░▒   ▒  ▒▒   ▓▒█░░ 
░ ░▒  ░ ░░ ░░   ░ ▒░  ▒   ▒▒ ░░ ░▒ ▒░ ░ ░  ░     ░   ░   ▒   ▒▒ ░░ 
░  ░  ░     ░   ░ ░   ░   ▒   ░ ░░ ░    ░      ░ ░   ░   ░   ▒     
      ░           ░       ░  ░░  ░      ░  ░         ░       ░  ░   
"""

    return title

# Animate Text in Line
def animate_text(stdscr, y_offset, x_offset, string, delay=0.01):
        
    temp_string = ""

    # Move in Text at Offset
    for letter in string:
        temp_string += letter
        stdscr.addstr(y_offset, x_offset, temp_string, curses.A_BOLD)  # Below title
        stdscr.refresh()
        time.sleep(delay)



# Function for Drawing the Main Menu
def draw_menu(stdscr):
    global MENU_OPEN

    # Text Variables
    subtitle_1 = "Welcome to Snake GA, the Genetic Algorithm Playground for Snake"
    subtitle_2 = "The Genetic Algorithm, right here in your console!"
    menu_title = "MENU"
    menu_option_1 = "Press '1' to Play a normal round of Snake, see what Fitness score you can achieve!"
    menu_option_2 = "Press '2' to Train the algorithm, and see how the Genetic Algorithm Learns to Play Snake!"
    menu_option_3 = "Press '3' to Enter the Playground, Test pre-trained Snakes or enter your own Weights and see how well the snake performs!"
    menu_option_4 = "Press 'q' to Quit, thank you for playing!"


    
    # Draw the title first
    title_text = get_snake_menu_text()
    stdscr.addstr(1, 1, title_text)
    stdscr.refresh()
    
    # Check if Animation should be played
    if not MENU_OPEN:

        # Animate Menu Text
        animate_text(stdscr, 11, 1, subtitle_1, 0.01,)
        animate_text(stdscr, 12, 1, subtitle_2, 0.01,)
        animate_text(stdscr, 14, 1, menu_title, 0.02)
        animate_text(stdscr, 15, 1, menu_option_1, 0.01,)
        animate_text(stdscr, 16, 1, menu_option_2, 0.01,)
        animate_text(stdscr, 17, 1, menu_option_3, 0.01,)
        animate_text(stdscr, 18, 1, menu_option_4, 0.01,)

        # Change Menu State to Open
        MENU_OPEN = True

    if MENU_OPEN:

        # Add Text
        stdscr.addstr(11, 1, subtitle_1, curses.A_BOLD)
        stdscr.addstr(12, 1, subtitle_2, curses.A_BOLD)
        stdscr.addstr(14, 1, menu_title, curses.A_BOLD)    
        stdscr.addstr(15, 1, menu_option_1, curses.A_BOLD)
        stdscr.addstr(16, 1, menu_option_2, curses.A_BOLD)
        stdscr.addstr(17, 1, menu_option_3, curses.A_BOLD)
        stdscr.addstr(18, 1, menu_option_4, curses.A_BOLD)

        # Blink Text
        stdscr.addstr(20, 1, "PRESS AN OPTION TO CONTINUE", curses.A_BOLD)
        stdscr.refresh()
        time.sleep(0.5)
        stdscr.addstr(20, 1, "                                  ", curses.A_BOLD)
        stdscr.refresh()
        time.sleep(0.5)

    
    
def run_game(stdscr):
    global GAME_SPEED

    # Track Game State
    active = True
    game_state = "Menu"
    prev_game_state = None

    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(1)  # Non-blocking input
    stdscr.timeout(100)  # Refresh rate (controls speed)

    snakeAI = SnakeAI()

    # Game Loop
    while active:

        if game_state == "Menu":
            global MENU_OPEN
            
            draw_menu(stdscr)
            key = stdscr.getch()

            # Catch Keytrokes
            if key == ord("1"):
                snake = Snake()
                game_state = "Running"
                prev_game_state = "Running"
                stdscr.refresh()

            elif key == ord("q"):
                stdscr.clear()
                animate_text(stdscr, 1, 1, "Goodbye!", 0.1)
                time.sleep(3)
                break
    

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

            key = stdscr.getch()
            if key == ord("w"):
                if GAME_SPEED > 0.001:
                    GAME_SPEED -= 0.001
            elif key == ord("s"):
                GAME_SPEED += 0.001  # Increase speed by 10% (slow down game)

            time.sleep(GAME_SPEED)  # Game speed

            if snake_status == "collision":
                game_state = "Game Over"


        # TODO: Playground Loop

    
        if game_state == "Game Over":

            # Game Over Screen
            stdscr.clear()
            stdscr.addstr(0, 0, "Game Over", curses.A_BOLD)
            stdscr.addstr(1, 0, 'Press "r" to try again')
            stdscr.addstr(2, 0, 'Press "m" to return to menu')
            stdscr.addstr(3, 0, 'Press "q" to quit')

            # Refresh Screen
            stdscr.refresh()

            # Catch Key Strokes
            key = stdscr.getch()
            if key == ord("r"):

                snake = Snake()
                snakeAI = SnakeAI()
                game_state = prev_game_state # Set Game to Previous Game State (Before Game Over)
                stdscr.clear()

            elif key == ord("m"):
                game_state = "Menu"
                stdscr.clear()


            elif key == ord("q"):
                break

            

if __name__ == "__main__":
    curses.wrapper(run_game)
