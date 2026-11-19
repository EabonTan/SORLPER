import sys
import gym
import torch
import random
import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from DDPGPER.prioritized_memory import Memory
from env1 import Env1
from Parameters import env_params, agent_params

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# DDPG Agent with PER
class DDPGAgent:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.max_action = 1.0

        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=agent_params['actor_lr'])

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=agent_params['critic_lr'])

        
        self.memory = Memory(agent_params['buffer_size'])
        self.batch_size = agent_params['batch_size']
        self.tau = agent_params['tau']
        self.discount = agent_params['discount_rate']

        
        self.noise_std = agent_params['noise_std']
        self.noise_decay = (agent_params['noise_std'] - agent_params['noise_std_min']) / num_episodes

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = (action + noise).clip(-self.max_action, self.max_action)

        return action

    def update(self):
        if self.memory.tree.n_entries < self.batch_size:
            return

        
        batch, idxs, is_weights = self.memory.sample(self.batch_size)
        is_weights = torch.FloatTensor(is_weights).to(self.device)

        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([b[1] for b in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([b[4] for b in batch])).to(self.device).unsqueeze(1)

        
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        current_Q = self.critic(states, actions)
        td_errors = torch.abs(current_Q - target_Q).squeeze(1).detach().cpu().numpy()
        critic_loss = (is_weights * F.mse_loss(current_Q, target_Q, reduction='none')).mean()

       
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        
        for i, idx in enumerate(idxs):
            self.memory.update(idx, td_errors[i])

       
        if self.noise_std > agent_params['noise_std_min']:
            self.noise_std -= self.noise_decay

    def save_model(self, num):
        torch.save(self.actor.state_dict(), f"{env_params['save_path']}/actor_{num}.pth")
        torch.save(self.critic.state_dict(), f"{env_params['save_path']}/critic_{num}.pth")

    def append_sample(self, state, action, reward, next_state, done):
        
        self.memory.add(1.0, (state, action, reward, next_state, done))


def save_dict_as_txt(dictionary, filename):
    with open(filename, 'w') as f:
        for key, value in dictionary.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    num_episodes = env_params['num_trials']
    save_path = env_params['save_path']
    save_dict_as_txt(env_params, f'{save_path}/env_params.txt')
    save_dict_as_txt(agent_params, f'{save_path}/agent_params.txt')
    num_repeats = env_params['num_repeats']

    for j in range(num_repeats):
        print(f'###### Round{j + 1} ######')
        env = Env1(j)
        state_size = env_params['state_dim']
        action_size = env_params['action_dim']
        max_steps = env_params['max_steps']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        agent = DDPGAgent(state_size, action_size, device)

        best_steps = np.inf
        return_list = []
        sequence = {}
        best_model = j

        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc=f'迭代 {i}') as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    state = env.reset()
                    episode_return = 0
                    done = False
                    step_count = 0
                    agent_sequence = np.array(env_params['agent_start']).reshape(1, 2)

                    while not done and step_count < max_steps:
                        action = agent.select_action(state)
                        next_state, reward, done, info = env.step(action)
                        agent_sequence = np.concatenate((agent_sequence, info['agent']), 0)

                        agent.append_sample(state, action, reward, next_state, done)
                        agent.update()

                        state = next_state
                        episode_return += reward
                        step_count += 1

                    return_list.append(episode_return)

                    
                    if info['arrive'] == 1 and step_count < best_steps:
                        best_steps = step_count
                        agent.save_model(best_model)

                    sequence[f'episode_{num_episodes / 10 * i + i_episode + 1}'] = agent_sequence

                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode': f'{num_episodes / 10 * i + i_episode + 1}',
                            'return': f'{np.mean(return_list[-10:]):.3f}'
                        })
                    pbar.update(1)

        
        np.savetxt(f'{save_path}/return_List_env{j}.txt', np.array(return_list))
        np.save(f'{save_path}/agent_sequence_env{j}.npy', sequence)
