import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.networks import Actor, Critic
from utils.memory_modules import ReplayBuffer
from utils.noise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(object):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, config):
        """
        Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            config (dict) : dictionary of hyper-parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.config = config

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, fc_units=[256, 128]).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc_units=[256, 128]).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.config["LR_ACTOR"])

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, fcs_units=[256], fc_units=[128]).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, fcs_units=[256], fc_units=[128]).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.config["LR_CRITIC"],
                                           weight_decay=self.config["WEIGHT_DECAY"])

        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        self.t_step = 0
        self.update_every = self.config["UPDATE_EVERY"]

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.config["BUFFER_SIZE"], self.config["BATCH_SIZE"], random_seed)

    def hard_copy_weights(self, target, source):
        # Copy weights from source to target network (part of initialization)
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory, and use random sample from buffer to learn.
        self.memory.add(state, action, reward, next_state, done)

        # If enough samples are available in memory, train models
        if len(self.memory) > self.config["BATCH_SIZE"]:
            experiences = self.memory.sample()
            self.learn(experiences, self.config["GAMMA"])

    def act(self, state, add_noise=True):
        # Returns noisy actions for given state as per current policy.
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        # Reset Noise Generator
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            t_step (int) : total_step count
            update_every (int) : update model once in every n steps
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip gradients of critic
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Clip gradients of actor
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        if self.t_step % self.update_every == 0:
            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target, self.config["TAU"])
            self.soft_update(self.actor_local, self.actor_target, self.config["TAU"])

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, name="agent_1_"):
        # Save actor and critic models
        torch.save(self.actor_local.state_dict(), f"{name}_actor_checkpoint_actor.pth")
        torch.save(self.critic_local.state_dict(), f"{name}_critic_checkpoint_actor.pth")