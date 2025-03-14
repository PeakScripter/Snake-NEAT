# Snake Game with NEAT AI
![image](https://github.com/user-attachments/assets/704e1615-7877-4ace-a04f-99b2987f079d)

This project implements a classic Snake game with an AI that learns to play using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. The AI evolves neural networks that control the snake, learning to navigate the game environment, avoid obstacles, and collect food efficiently.

## Description

The Snake Game AI project combines:

1. A fully functional Snake game built with Pygame
2. NEAT algorithm implementation for training neural networks
3. Visualization tools to track the AI's learning progress

The neural networks receive information about the snake's current state (direction, dangers, food location) and output movement decisions. Through evolutionary processes, the networks improve over generations, developing strategies to maximize score.

## Features

- **Classic Snake Game**: Complete implementation with customizable parameters
- **NEAT AI Training**: Evolutionary algorithm that develops neural networks to play the game
- **Visualization**: Tools to visualize network structures and training statistics
- **Headless Mode**: Option to run training without visual rendering for faster evolution
- **Save/Load Models**: Ability to save trained networks and load them for testing

## Requirements

- Python 3.6+
- pygame
- neat-python
- numpy
- matplotlib (for visualization)

## Installation

```bash
# Clone the repository
git clone https://github.com/peakscripter/snake-NEAT.git

# Install required packages
pip install pygame neat-python numpy matplotlib
```

## Usage

### Training a new AI

```bash
python main.py
```

This will start the training process with the NEAT algorithm. The training will run for 100 generations by default, and the best performing neural network will be saved as `winner.pkl`.

### Testing a pre-trained AI

To test a previously trained neural network:

1. Set `train_mode = False` in the main section of `main.py`
2. Run `python main.py`

## How It Works

### Game Environment

The Snake game provides the environment for the AI to learn in. The snake must navigate a grid, collecting food while avoiding walls and its own body.

### Neural Network Input

The neural network receives 14 inputs:
- Current direction (4 inputs, one-hot encoded)
- Danger detection in each direction (4 inputs)
- Food location relative to snake (4 inputs)
- Normalized distance to food (2 inputs)

### Neural Network Output

The network outputs 3 possible actions:
- Continue straight
- Turn right
- Turn left

### Fitness Function

The AI is rewarded for:
- Collecting food (+10 points + bonus based on snake length)
- Moving closer to food (+0.5)
- Surviving longer (small bonus per step)

And penalized for:
- Hitting walls or itself (-10)
- Moving away from food (-0.5)
- Taking too long to find food (-10 after max steps)

## Configuration

The NEAT algorithm's parameters are defined in `config-neat.txt`. You can modify these parameters to adjust:
- Population size
- Mutation rates
- Species threshold
- Network architecture constraints
![image](https://github.com/user-attachments/assets/582ea30a-2484-4c44-8f7b-9ae36bb928e4)

## License

[MIT License](LICENSE)

## Acknowledgments

- The NEAT algorithm was developed by Kenneth O. Stanley and Risto Miikkulainen
- This implementation uses the neat-python library
