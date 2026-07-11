# AI-Powered Turn-Based Strategy

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Pygame](https://img.shields.io/badge/Pygame-2.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A high-performance, AI-driven Turn-Based Strategy (TBS) game built from scratch using **Python** and **PyTorch**. This project demonstrates the implementation of a state-of-the-art **AlphaZero-style reinforcement learning agent** capable of mastering complex strategic gameplay through self-play.

## Key Features

*   **AI Agent**: Implements a Deep ResNet model combined with Monte Carlo Tree Search (MCTS), inspired by DeepMind's AlphaZero.
*   **High-Performance Engine**: Game logic is optimized using **Numba** (JIT compilation), enabling thousands of simulations per second for rapid training.
*   **Interactive Visualization**: A clean, responsive GUI built with **Pygame** allows for real-time observation of AI matches or Human vs. AI gameplay.
*   **Training Pipeline**: Includes a multi-process training system with experience replay, symmetry augmentation, and adaptive learning rate scheduling.
*   **TensorBoard Integration**: Real-time monitoring of training metrics (loss, win rates, game length).

## The Game

The game is a 1x1 tactical strategy played on a 10x10 grid.

*   **Goal**: Eliminate all enemy units or control the most territory by the end of 50 turns.
*   **Economy**: Capture territory to generate coins.
*   **Units**:
    *   **Peasant**: Cheap, weak, captures territory. Can merge to form Warriors.
    *   **Warrior**: Expensive, strong, destroys enemies.
*   **Mechanics**:
    *   **Move**: Units can move to adjacent cells.
    *   **Attack**: Warriors automatically destroy enemies upon moving into their space.
    *   **Merge**: Two peasants can merge into a Warrior.
    *   **Upkeep**: Units require gold maintenance every turn.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rerqaa/AI_TBS.git
    cd AI_TBS
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: A CUDA-capable GPU is highly recommended for training the AI.*

## Usage

### Play the Game
Run the main application to open the menu. You can choose between **AI vs AI**, **Human vs AI**, or **Human vs Human**.

```bash
python main.py
```

### Train the AI
To start the self-play training loop:

```bash
python train.py
```
*   The script will automatically use all available CPU cores for self-play data generation and the GPU for model updates.
*   Checkpoints are saved as `latest_checkpoint.pth` and the best performing model as `best_model.pth`.
*   Logs are saved to the `runs/` directory.

### Monitor Training
Visualize training progress with TensorBoard:

```bash
tensorboard --logdir=runs
```

## Technical Deep Dive

### Neural Network Architecture (`model.py`)
The core of the AI is a **Residual Neural Network (ResNet)**:
*   **Input**: $6 \times 10 \times 10$ spatial tensor (Unit positions, Territory control) + Global features (Coin counts).
*   **Backbone**: Initial Convolution layer followed by **10 Residual Blocks**.
*   **Heads**:
    *   **Policy Head**: Outputs a probability distribution over all possible moves (Move, Build, Merge).
    *   **Value Head**: Predicts the win probability ($[-1, 1]$) for the current state.

### Monte Carlo Tree Search (MCTS) (`mcts.py`)
The agent uses MCTS to look ahead and evaluate future states.
*   **Selection**: Uses the PUCT (Predictor + Upper Confidence Bound applied to Trees) formula to balance exploration and exploitation.
*   **Evaluation**: Leaf nodes are evaluated by the Neural Network.
*   **Backpropagation**: Values are propagated up the tree to update Q-values.

### Optimization (`game_state.py`)
To handle the computational load of MCTS (which requires thousands of game state simulations), the core game logic is written with **Numba's `@njit` decorator**. This compiles Python code to optimized machine code, resulting in a **100x-1000x speedup** compared to standard Python, making Python a viable language for this compute-intensive task.

## Project Structure

*   `main.py`: Entry point for the GUI application.
*   `train.py`: Main training loop (Self-Play -> Training -> Evaluation).
*   `game_state.py`: Core game logic and rules (Numba-optimized).
*   `model.py`: PyTorch definition of the ResNet architecture.
*   `mcts.py`: Implementation of the Monte Carlo Tree Search.
*   `visualizer.py`: Pygame rendering logic.
*   `bots.py`: Wrapper classes for AI and Random agents.
*   `constants.py`: Game configuration (Board size, Costs, etc.).

## Future Improvements

*   [ ] Add more unit types and terrain features.
*   [ ] Export model to ONNX for web-based inference.
*   [ ] Visualize MCTS search tree in real-time.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
