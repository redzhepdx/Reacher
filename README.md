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
pip install -r requirements.txt
```

### Training an Agent
- The last kernel trains in `Reacher.ipynb` notebook trains agent until it reaches the target score. It is mandatory to run all previous kernels to load environment and instantiate a DDPG agent.
- Optional : You can change the learning algorithm's hyper-parameters by updating the config dictionary in the notebook

#### Initial Config
```
config = {
    "BUFFER_SIZE" : int(1e6), # replay buffer size
    "BATCH_SIZE" : 512,       # minibatch size
    "GAMMA" : 0.99,           # discount factor
    "TAU" : 1e-3,             # for soft update of target parameters
    "LR_ACTOR" : 1e-4,        # learning rate of the actor 
    "LR_CRITIC" : 2e-4,       # learning rate of the critic
    "WEIGHT_DECAY" : 0,       # L2 weight decay
    "UPDATE_EVERY" : 3        # Soft Update Rate
}
```

### Project Structure and Instructions
- `agents/` -> Contains the implementations of DDPG Agent
- `models/` -> Contains the implementations of Critic and Actor Neural Networks. [Pytorch]
- `utils/` -> Contains memory modules and noise generator class implementations.
- `Reacher.ipynb` -> Execution of the algorithm. Training Agents and Unity Visualizations
- `report.pdf` -> description of the methods and application
- `*.pth files` -> pre-trained actor and critic models of the agent [agent_1_actor_checkpoint.pth, agent_1_critic_checkpoint.pth]