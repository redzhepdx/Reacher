[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


# Project 2: Continuous Control
Udacity Deep Reinforcement Learning Project 2 - Continuous Control with DDPG

### Introduction

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Setup
The project uses Jupyter Notebook. This command needs to be run to install the needed packages:
```
!pip -q install ./python
```

### Project Structure and Instructions
- `agents/` -> Contains the implementations of DDPG Agent
- `models/` -> Contains the implementations of Critic and Actor Neural Networks. [Pytorch]
- `utils/` -> Contains memory modules and noise generator class implementations.
- `Reacher.ipynb` -> Execution of the algorithm. Training Agents and Unity Visualizations
- `report.pdf` -> description of the methods and application
- `*.pth files` -> pre-trained actor and critic models of the agent [agent_1_actor_checkpoint.pth, agent_1_critic_checkpoint.pth]