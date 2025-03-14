import pygame
import time
import random
import numpy as np
import os
import neat
import math
import pickle
import warnings
import copy  # Missing import needed for the visualize module

# Initialize pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (213, 50, 80)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)

# Set display dimensions
DISPLAY_WIDTH = 200
DISPLAY_HEIGHT = 200

# Snake block size and speed
SNAKE_BLOCK = 10
SNAKE_SPEED = 100  # Increased for AI training

# Font styles
FONT_STYLE = pygame.font.SysFont("bahnschrift", 25)
SCORE_FONT = pygame.font.SysFont("comicsansms", 35)

class SnakeGame:
    def __init__(self, display=None, headless=False):
        self.headless = headless
        self.display_width = DISPLAY_WIDTH
        self.display_height = DISPLAY_HEIGHT
        self.snake_block = SNAKE_BLOCK
        
        # Initialize display if not headless
        if not headless:
            if display is None:
                self.display = pygame.display.set_mode((self.display_width, self.display_height))
                pygame.display.set_caption('Snake Game - NEAT AI')
            else:
                self.display = display
            self.clock = pygame.time.Clock()
        else:
            self.display = display  # Still store display reference even in headless mode
        
        # Initialize game state
        self.reset()

    def reset(self):
        """Reset the game state for a new game"""
        # Initialize snake position (middle of the screen)
        self.x1 = self.display_width / 2
        self.y1 = self.display_height / 2
        
        # Initialize change in position
        self.x1_change = 0
        self.y1_change = 0
        
        # Initialize snake
        self.snake_list = []
        self.snake_head = [self.x1, self.y1]
        self.snake_list.append(self.snake_head[:])
        self.length_of_snake = 5
        
        # Initialize food position
        self.foodx = round(random.randrange(0, self.display_width - self.snake_block) / 10.0) * 10.0
        self.foody = round(random.randrange(0, self.display_height - self.snake_block) / 10.0) * 10.0
        
        # Game state
        self.game_over = False
        self.steps_without_food = 0
        self.max_steps_without_food = (self.display_width * self.display_height) // (self.snake_block * self.snake_block)
        self.score = 0
        self.frames = 0
        self.direction = "RIGHT"
        
        return self.get_state()

    def get_state(self):
        """Get the current state of the game for the neural network"""
        # Direction of snake's head (one-hot encoded)
        dir_left = 1 if self.direction == "LEFT" else 0
        dir_right = 1 if self.direction == "RIGHT" else 0
        dir_up = 1 if self.direction == "UP" else 0
        dir_down = 1 if self.direction == "DOWN" else 0
        
        # Food position relative to snake head
        food_left = 1 if self.foodx < self.x1 else 0
        food_right = 1 if self.foodx > self.x1 else 0
        food_up = 1 if self.foody < self.y1 else 0
        food_down = 1 if self.foody > self.y1 else 0
        
        # Normalized distance to food
        food_distance_x = (self.foodx - self.x1) / self.display_width
        food_distance_y = (self.foody - self.y1) / self.display_height
        
        # Danger detection (check if there's a wall or body part in each direction)
        # Check left
        danger_left = 0
        if self.x1 - self.snake_block < 0:  # Wall on the left
            danger_left = 1
        else:
            for segment in self.snake_list[:-1]:
                if segment[0] == self.x1 - self.snake_block and segment[1] == self.y1:
                    danger_left = 1
                    break
                    
        # Check right
        danger_right = 0
        if self.x1 + self.snake_block >= self.display_width:  # Wall on the right
            danger_right = 1
        else:
            for segment in self.snake_list[:-1]:
                if segment[0] == self.x1 + self.snake_block and segment[1] == self.y1:
                    danger_right = 1
                    break
                    
        # Check up
        danger_up = 0
        if self.y1 - self.snake_block < 0:  # Wall above
            danger_up = 1
        else:
            for segment in self.snake_list[:-1]:
                if segment[0] == self.x1 and segment[1] == self.y1 - self.snake_block:
                    danger_up = 1
                    break
                    
        # Check down
        danger_down = 0
        if self.y1 + self.snake_block >= self.display_height:  # Wall below
            danger_down = 1
        else:
            for segment in self.snake_list[:-1]:
                if segment[0] == self.x1 and segment[1] == self.y1 + self.snake_block:
                    danger_down = 1
                    break
        
        # Return state vector
        return [
            dir_left, dir_right, dir_up, dir_down,
            danger_left, danger_right, danger_up, danger_down,
            food_left, food_right, food_up, food_down,
            food_distance_x, food_distance_y
        ]

    def step(self, action):
        """Take a step in the game with the given action
        Actions: 0 = straight, 1 = right turn, 2 = left turn
        Returns: (new_state, reward, done)
        """
        self.frames += 1
        reward = 0
        
        # Convert action to direction change
        if action == 0:  # Continue straight
            pass  # Keep the current direction
        elif action == 1:  # Turn right
            if self.direction == "UP":
                self.direction = "RIGHT"
            elif self.direction == "RIGHT":
                self.direction = "DOWN"
            elif self.direction == "DOWN":
                self.direction = "LEFT"
            elif self.direction == "LEFT":
                self.direction = "UP"
        elif action == 2:  # Turn left
            if self.direction == "UP":
                self.direction = "LEFT"
            elif self.direction == "LEFT":
                self.direction = "DOWN"
            elif self.direction == "DOWN":
                self.direction = "RIGHT"
            elif self.direction == "RIGHT":
                self.direction = "UP"
        
        # Set movement based on direction
        if self.direction == "LEFT":
            self.x1_change = -self.snake_block
            self.y1_change = 0
        elif self.direction == "RIGHT":
            self.x1_change = self.snake_block
            self.y1_change = 0
        elif self.direction == "UP":
            self.y1_change = -self.snake_block
            self.x1_change = 0
        elif self.direction == "DOWN":
            self.y1_change = self.snake_block
            self.x1_change = 0
        
        # Update snake position
        self.x1 += self.x1_change
        self.y1 += self.y1_change
        
        # Check for boundary collision
        if (self.x1 >= self.display_width or self.x1 < 0 or 
            self.y1 >= self.display_height or self.y1 < 0):
            reward = -10  # Penalty for hitting the wall
            self.game_over = True
            return self.get_state(), reward, True
        
        # Update snake
        self.snake_head = [self.x1, self.y1]
        self.snake_list.append(self.snake_head[:])
        
        # Calculate distance to food (before and after movement)
        new_dist = math.sqrt((self.x1 - self.foodx)**2 + (self.y1 - self.foody)**2)
        prev_dist = math.sqrt((self.x1 - self.x1_change - self.foodx)**2 + 
                              (self.y1 - self.y1_change - self.foody)**2)
        
        # Check if food eaten
        if abs(self.x1 - self.foodx) < self.snake_block and abs(self.y1 - self.foody) < self.snake_block:
            # Generate new food position
            self.foodx = round(random.randrange(0, self.display_width - self.snake_block) / 10.0) * 10.0
            self.foody = round(random.randrange(0, self.display_height - self.snake_block) / 10.0) * 10.0
            
            # Avoid placing food on snake
            food_on_snake = True
            while food_on_snake:
                food_on_snake = False
                for segment in self.snake_list:
                    if abs(segment[0] - self.foodx) < self.snake_block and abs(segment[1] - self.foody) < self.snake_block:
                        self.foodx = round(random.randrange(0, self.display_width - self.snake_block) / 10.0) * 10.0
                        self.foody = round(random.randrange(0, self.display_height - self.snake_block) / 10.0) * 10.0
                        food_on_snake = True
                        break
            
            # Increase snake length and reset steps counter
            self.length_of_snake += 1
            self.steps_without_food = 0
            self.score += 1
            reward = 10 + (self.length_of_snake * 2)  # Increasing reward for food as snake grows
        else:
            # Remove extra segments if snake hasn't grown
            if len(self.snake_list) > self.length_of_snake:
                del self.snake_list[0]
            
            # Penalize for not finding food
            self.steps_without_food += 1
            if self.steps_without_food > self.max_steps_without_food:
                reward = -10  # Penalty for taking too long
                self.game_over = True
                return self.get_state(), reward, True
        
        # Reward for getting closer to food
        if new_dist < prev_dist:
            reward += 0.5  # Increased reward for moving towards food
        else:
            reward -= 0.5  # Increased penalty for moving away from food
        
        # Add a small penalty for each step to encourage efficiency
        reward -= 0.01
        
        # Check for self collision
        for segment in self.snake_list[:-1]:
            if segment[0] == self.snake_head[0] and segment[1] == self.snake_head[1]:
                reward = -10  # Penalty for colliding with self
                self.game_over = True
                return self.get_state(), reward, True
        
        # Update display if not headless
        if not self.headless and self.display is not None:
            self.render()
            self.clock.tick(SNAKE_SPEED)
        
        return self.get_state(), reward, self.game_over

    def render(self):
        """Draw the game state on the screen"""
        # Draw background
        self.display.fill(BLACK)
        
        # Draw food
        pygame.draw.rect(self.display, RED, [self.foodx, self.foody, self.snake_block, self.snake_block])
        
        # Draw snake
        for segment in self.snake_list:
            pygame.draw.rect(self.display, GREEN, [segment[0], segment[1], self.snake_block, self.snake_block])
        
        # Draw score
        value = SCORE_FONT.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(value, [0, 0])
        
        # Update display
        pygame.display.update()

# NEAT implementation functions
def eval_genomes(genomes, config):
    """Evaluates each genome by letting it play the game"""
    # Create a display window if visualizing
    display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption('Snake Game - NEAT AI Training')

    for genome_id, genome in genomes:
        # Initialize the neural network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Initialize fitness
        genome.fitness = 0
        
        # Run multiple games for each genome to get better evaluation
        num_games = 3
        total_fitness = 0
        
        for _ in range(num_games):
            # Initialize game environment
            game = SnakeGame(display=display, headless=False)  # Change to False to see the game
            state = game.reset()
            
            # Game loop for this genome
            steps = 0
            max_steps = 100  # Reduced from 300 to prevent getting stuck
            
            while not game.game_over and steps < max_steps:
                # Check for quit event
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                
                # Get network output
                output = net.activate(state)
                action = np.argmax(output)  # Choose action with highest output
                
                # Take a step in the environment
                state, reward, done = game.step(action)
                
                # Update fitness
                total_fitness += reward
                
                # Break if game over
                if done:
                    break
                    
                steps += 1
            
            # Add a small bonus for surviving longer
            total_fitness += steps * 0.1
        
        # Set genome fitness (ensure it's never negative to keep genomes in the population)
        genome.fitness = max(0.1, total_fitness / num_games)
        print(f"Genome {genome_id} fitness: {genome.fitness}")

def run_neat(config_path):
    """Runs the NEAT algorithm to train a neural network to play Snake"""
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Create the population
    p = neat.Population(config)
    
    # Add a reporter to show progress
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    
    # Run for up to 50 generations initially
    winner = p.run(eval_genomes, 100)
    
    # Save the winner
    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    # Show final stats
    print('\nBest genome:\n{!s}'.format(winner))
    
    # Visualize the stats and the network
    try:
        import visualize
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
        
        # Show the network
        node_names = {
            -1: 'Dir_Left', -2: 'Dir_Right', -3: 'Dir_Up', -4: 'Dir_Down',
            -5: 'Danger_Left', -6: 'Danger_Right', -7: 'Danger_Up', -8: 'Danger_Down',
            -9: 'Food_Left', -10: 'Food_Right', -11: 'Food_Up', -12: 'Food_Down',
            -13: 'Food_Dist_X', -14: 'Food_Dist_Y',
            0: 'Straight', 1: 'Right Turn', 2: 'Left Turn'
        }
        visualize.draw_net(config, winner, True, node_names=node_names)
    except Exception as e:
        print(f"Visualization error: {e}")
    
    return winner

def test_best_network(config_path, genome_path='winner.pkl'):
    """Test the best network in the game"""
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Load the winner genome
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    # Create the neural network
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Initialize game
    game = SnakeGame(headless=False)
    state = game.reset()
    
    # Game loop
    while not game.game_over:
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Get network output
        output = net.activate(state)
        action = np.argmax(output)
        
        # Take a step in the environment
        state, _, _ = game.step(action)
        
        # Add a small delay to make it watchable
        pygame.time.delay(50)
        
        # Force display update
        pygame.display.update()
    
    # Game over
    print("Final Score:", game.score)
    pygame.time.delay(2000)  # Wait 2 seconds before quitting
    pygame.quit()

# Main function
if __name__ == "__main__":
    try:
        import visualize
    except ImportError:
        print("Warning: visualize module could not be imported")
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat.txt')
    
    # Either run training or test a pre-trained network
    train_mode = True
    
    if train_mode:
        winner = run_neat(config_path)
        # Test the winner after training
        test_best_network(config_path)
    else:
        # Test a pre-trained network
        test_best_network(config_path)