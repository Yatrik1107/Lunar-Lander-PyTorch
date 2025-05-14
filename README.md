# Lunar Lander with Dueling Double DQN

![Lunar Lander Environment](https://gymnasium.farama.org/_images/lunar_lander.gif)

## Project Overview

This project implements an advanced Deep Reinforcement Learning agent to solve OpenAI Gymnasium's Lunar Lander environment. The agent combines two powerful DQN enhancements:

1. **Dueling DQN Architecture**: Separates state value and action advantage estimation
2. **Double DQN**: Reduces overestimation bias in Q-learning by decoupling action selection and evaluation

The implementation successfully trains an agent to master the challenging Lunar Lander environment, achieving an average score of 250+ over 100 consecutive episodes.

## Architecture

### Dueling DQN Network

The core innovation in this implementation is the Dueling DQN architecture, which splits the traditional Q-network into two streams:

- **Value Stream**: Estimates how good it is to be in a given state (V(s))
- **Advantage Stream**: Estimates the relative advantage of each action in that state (A(s,a))

These streams are combined using the formula:
```
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
```

This separation allows the network to learn which states are valuable without having to learn the effect of each action at each state, significantly improving learning efficiency and stability.

### Double DQN Implementation

To address the overestimation bias in traditional Q-learning, this implementation uses Double DQN:

1. The policy network selects the best action
2. The target network evaluates that action

This decoupling prevents the same network from both selecting and evaluating actions, reducing the tendency to overestimate Q-values and improving learning stability.

## Key Features

- **Experience Replay Buffer**: Stores transitions for efficient learning and breaks correlations in sequential data
- **Epsilon-Greedy Exploration**: Balanced exploration-exploitation with decay mechanism
- **Target Network**: Stabilizes training with periodic updates
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Adaptive learning rate during training
- **Reward Clipping**: Limits reward magnitude to improve stability
- **Early Stopping**: Detects when environment is solved to optimize training time
- **Model Checkpointing**: Saves best-performing models during training

## Requirements

- Python 3.6+
- PyTorch
- Gymnasium
- NumPy

## Installation

```bash
# Clone this repository
git clone https://github.com/Yatrik1107/Lunar-Lander-PyTorch.git
cd dueling-double-dqn-lunar-lander

# Install dependencies
pip install torch gymnasium numpy
```

## Usage

### Training the Agent

```python
# Set pre_existing_model to False to train from scratch
pre_existing_model = False
python Dueling_DQN_with_Lunar_Lander.py
```

### Loading a Pre-trained Model

```python
# Set pre_existing_model to True and specify model name
pre_existing_model = True
model_name = "best_policy_net.pth"
python Dueling_DQN_with_Lunar_Lander.py
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| EPISODES | 1000 | Maximum number of training episodes |
| MAIN_BUFFER_MEMORY | 200000 | Size of experience replay buffer |
| MIN_BUFFER_MEMORY | 10000 | Minimum buffer size before starting learning |
| BATCH_SIZE | 64 | Number of samples per learning update |
| GAMMA | 0.99 | Discount factor for future rewards |
| TARGET_UPDATE | 5 | Frequency of target network updates |
| EPSILON_START | 1.0 | Initial exploration rate |
| EPSILON_END | 0.01 | Final exploration rate |
| EPSILON_DECAY | 0.99 | Exploration decay factor |
| LEARNING_RATE | 0.0003 | Learning rate for optimizer |

## Results

The agent successfully solves the environment (achieves an average score of 250+ over 100 consecutive episodes) in 715 episodes with a training time of approximately 21 minutes, demonstrating the effectiveness of the Dueling Double DQN approach.

Training progress shows steady improvement in performance:

```
Episode : 100 / 1000 Reward : -422.54 Epsilon : 0.36603 Average Reward : -337.07986
Episode : 200 / 1000 Reward : -179.79 Epsilon : 0.13398 Average Reward : -235.07872
Episode : 300 / 1000 Reward : -123.38 Epsilon : 0.04904 Average Reward : -12.26667
Episode : 400 / 1000 Reward : 35.64 Epsilon : 0.01795 Average Reward : 171.81070
Episode : 600 / 1000 Reward : 288.37 Epsilon : 0.01000 Average Reward : 156.69277
Episode : 700 / 1000 Reward : 253.53 Epsilon : 0.01000 Average Reward : 237.27635
Environment is solved in 715 episodes! Average Reward : 250.45319
```

Evaluation results over 10 episodes show consistent performance:

```
Episode : 1 / 10 Reward : 273.12
Episode : 2 / 10 Reward : 233.56
Episode : 3 / 10 Reward : 282.41
Episode : 4 / 10 Reward : 266.11
Episode : 5 / 10 Reward : 122.12
Episode : 6 / 10 Reward : 244.18
Episode : 7 / 10 Reward : 262.36
Episode : 8 / 10 Reward : 225.73
Episode : 9 / 10 Reward : 267.67
Episode : 10 / 10 Reward : 274.47
```

The average evaluation reward of 245.17 demonstrates the agent's ability to consistently perform well in the Lunar Lander environment.

## Technical Implementation Details

### Network Architecture
- **Feature Extractor**: 2 fully connected layers (8→256→256)
- **Value Stream**: 2 fully connected layers (256→128→1)
- **Advantage Stream**: 2 fully connected layers (256→128→4)

### Optimization
- **Optimizer**: Adam with initial learning rate of 0.0003
- **Learning Rate Schedule**: Linear decay from 1.0 to 0.5 across training
- **Loss Function**: Mean Squared Error between current and target Q-values
- **Gradient Clipping**: Maximum norm of 10 to prevent gradient explosions

## Acknowledgements

- [OpenAI Gymnasium](https://gymnasium.farama.org/) for the Lunar Lander environment
- [DeepMind's paper on Dueling DQN](https://arxiv.org/abs/1511.06581)
- [Google DeepMind's paper on Double DQN](https://arxiv.org/abs/1509.06461)

## Author

Yatrik N. Patel

---

*This implementation demonstrates advanced deep reinforcement learning techniques for solving complex control tasks. Feel free to use this code as a reference for your own reinforcement learning projects.*
