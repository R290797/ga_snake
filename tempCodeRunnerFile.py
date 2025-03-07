# Collision Detection (Wall or Self)
    if new_head in snake or not (0 <= new_head[0] < WIDTH and TOP_BAR_HEIGHT <= new_head[1] < HEIGHT + TOP_BAR_HEIGHT):
        running = False  # Game Over