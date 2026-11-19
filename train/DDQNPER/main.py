import random
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from batch_env1 import Env1
from batch_Parameters import agent_params, env_params
from PER.prioritized_memory import Memory

# Training Parameters
num_episodes = env_params['num_trials']
DISCOUNT = agent_params['discount_rate']
LEARNING_RATE = agent_params['learning_rate']
MEMORY_SIZE = agent_params['buffer_size']
EPSILON_START = agent_params['start_epsilon']
EPSILON_END = agent_params['final_epsilon']
BATCH_SIZE = agent_params['batch_size']
TRAIN_START = agent_params['minimal_size']

class DQN(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent():
    def __init__(self, state_size, action_size, device):
        self.render = False
        self.load_model = False
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = DISCOUNT
        self.learning_rate = LEARNING_RATE
        self.memory_size = MEMORY_SIZE
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / num_episodes
        self.batch_size = BATCH_SIZE
        self.train_start = TRAIN_START

        self.memory = Memory(self.memory_size)
        self.model = DQN(state_size, action_size, device).to(device)
        self.target_model = DQN(state_size, action_size, device).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state).float().to(device)
            with torch.no_grad():
                q_value = self.model(state)
                _, action = torch.max(q_value, 1)
            return int(action)

    def save_model(self, num):
        torch.save(self.model.state_dict(), f"{env_params['save_path']}/Q_net{num}.pth")

    def append_sample(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).to(device)
        next_state_tensor = torch.FloatTensor(next_state).to(device)

        with torch.no_grad():
            current_qs = self.model(state_tensor)
            next_q_online = self.model(next_state_tensor)
            next_action = torch.argmax(next_q_online, dim=1)
            next_q_target = self.target_model(next_state_tensor)
            next_q = next_q_target.gather(1, next_action.unsqueeze(1)).squeeze(1)

            if not done:
                new_q = reward + self.discount_factor * next_q
            else:
                new_q = torch.tensor(reward, dtype=torch.float32, device=device)

            td_error = torch.abs(new_q - current_qs[0][action]).cpu()

        self.memory.add(td_error, (state, action, reward, next_state, done))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)

        if mini_batch[-1] == 0 and len(mini_batch) > 1:
            mini_batch[-1] = mini_batch[-2]

        mini_batch = np.array(mini_batch, dtype=object)

        states = np.array([ss[0] for ss in mini_batch]).reshape(self.batch_size, self.state_size)
        actions = np.array([ss[1] for ss in mini_batch])
        rewards = np.array([ss[2] for ss in mini_batch])
        next_states = np.array([ss[3] for ss in mini_batch]).reshape(self.batch_size, self.state_size)
        dones = np.array([ss[4] for ss in mini_batch]).astype(int)

        states = torch.Tensor(states).float().to(device)
        next_states = torch.Tensor(next_states).float().to(device)
        actions = torch.LongTensor(actions).view(-1, 1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        current_q_values = self.model(states)
        current_q = current_q_values.gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_online = self.model(next_states)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target_model(next_states)
            next_q_values = next_q_target.gather(1, next_actions).squeeze(1)
            target = rewards + (1 - dones) * self.discount_factor * next_q_values

        errors = torch.abs(current_q - target).detach().cpu().numpy()

        for i in range(self.batch_size):
            self.memory.update(idxs[i], errors[i])

        is_weights_tensor = torch.FloatTensor(is_weights).to(device)
        elementwise_loss = F.mse_loss(current_q, target, reduction='none')
        loss = (is_weights_tensor * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

def save_dict_as_txt(dictionary, filename):
    with open(filename, 'w') as f:
        for key, value in dictionary.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    save_path = env_params['save_path']
    save_dict_as_txt(env_params, f'{save_path}/env_params.txt')
    save_dict_as_txt(agent_params, f'{save_path}/agent_params.txt')

    num_repeats = env_params['num_repeats']

    for j in range(num_repeats):
        print(f'######Round{j + 1}######')
        env = Env1(j)

        state_size = env_params['state_dim']
        action_size = env_params['action_dim']
        max_steps = env_params['max_steps']

        device = torch.device("cpu")
        print(f"Using device: {device}")

        agent = DQNAgent(state_size, action_size, device)
        best_steps = np.inf
        return_list = []
        sequence = {}
        best_model = j

        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    done = False
                    step = 0
                    arrive_num = 0
                    episode_return = 0

                    agent_sequence = np.array(env_params['agent_start']).reshape(1, 2)
                    state = env.reset()
                    state = np.reshape(state, [1, state_size])

                    while not done:
                        step += 1
                        action = agent.get_action(state)
                        next_state, reward, done, info = env.step(action)
                        next_state = np.reshape(next_state, [1, state_size])

                        agent_sequence = np.concatenate((agent_sequence, info['agent']), 0)

                        if len(agent_sequence) >= max_steps:
                            done = True
                            reward = -1

                        episode_return += reward
                        agent.append_sample(state, action, reward, next_state, done)

                        if agent.memory.tree.n_entries >= agent.train_start:
                            agent.train_model()

                        state = next_state

                    if info['arrive'] == 1 and len(agent_sequence) < best_steps:
                        best_steps = len(agent_sequence)
                        agent.save_model(best_model)

                    agent.update_target_model()
                    sequence['episode' + str(num_episodes / 10 * i + i_episode + 1)] = agent_sequence
                    return_list.append(episode_return)

                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                            'return': '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)

        np.savetxt(f'{save_path}/return_List_env{j}.txt', np.array(return_list))
        np.save(f'{save_path}/agent_sequence_env{j}.npy', sequence)
