import sys
import gym
import torch
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from BETDQN.prioritized_memory import Memory
from batch_env1 import Env1
from batch_Parameters import env_params, agent_params


""" Training Parameters """
num_episodes = env_params['num_trials']
DISCOUNT = agent_params['discount_rate']
LEARNING_RATE = agent_params['learning_rate']
MEMORY_SIZE = agent_params['buffer_size']
EPSILON_START = agent_params['start_epsilon']     
EPSILON_END = agent_params['final_epsilon']     
BATCH_SIZE = agent_params['batch_size']   
TRAIN_START = agent_params['minimal_size']   
W1 = 0.2
W2 = 0.8

""" Feedforward network as the Q function """
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

"""
We define a class for the agent. 
The agent uses prioritized replay memory borrowed from: https://github.com/rlcode/per/tree/master
to search for the prioritized samples by means of the proposed BETDQNet prioritization score.
"""
class DQNAgent():
    def __init__(self, state_size, action_size, device):

        self.render = False
        self.load_model = False

        """ get the size of state and action spaces """
        self.state_size = state_size
        self.action_size = action_size

        """ training hyperparameters """
        self.discount_factor = DISCOUNT
        self.learning_rate = LEARNING_RATE
        self.memory_size = MEMORY_SIZE
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / num_episodes  
        self.batch_size = BATCH_SIZE
        self.train_start = TRAIN_START

        """ W1 is assigned to the TD error and W2 to the BE """
        self.w1 = W1
        self.w2 = W2
        self.td_buffer = deque(maxlen=self.memory_size)
        self.be_buffer = deque(maxlen=self.memory_size)

        self.memory = Memory(self.memory_size)

        self.model = DQN(state_size, action_size, device).to(device)
        self.target_model = DQN(state_size, action_size, device).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.update_target_model()


    def update_target_model(self):
        """ to periodically update the target model """
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        """ the agent follows an epsilon-greedy exploration """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state).float().to(device)
            # state = Variable(state).float().cpu()
            q_value = self.model(state)
            _, action = torch.max(q_value, 1)
            return int(action)

    def save_model(self, num):
        torch.save(self.model.state_dict(), f"{env_params['save_path']}/Q_net{num}.pth")

    def append_sample(self, state, action, reward, next_state, done):
        """ each sample to be appended in the replay memory, will be accompanied by its associated weighted prioritization score
                                                                    based on the TD and BE with designated W1 and W2 weights """

        # current_qs = self.model(Variable(torch.FloatTensor(state))).data
        # future_qs = self.target_model(Variable(torch.FloatTensor(next_state))).data
        
        state_tensor = torch.FloatTensor(state).to(device)  
        next_state_tensor = torch.FloatTensor(next_state).to(device) 

        current_qs = self.model(state_tensor).data
        future_qs = self.target_model(next_state_tensor).data

        if not done:
            max_future_q = torch.max(future_qs)
            new_q = reward + self.discount_factor * max_future_q
        else:
            new_q = reward

        """ here we construct the prioritization score to be used for transitions rankings """
        td_error = torch.abs(new_q - current_qs[0][action])
        be_error = torch.abs(new_q - current_qs)
        mean_be_error = torch.mean(be_error)
        # weighted_error = self.w1 * td_error + self.w2 * mean_be_error
        #
        # self.memory.add(weighted_error, (state, action, reward, next_state, done)) #torch.ones(1,1)
        
        weighted_error = (self.w1 * td_error + self.w2 * mean_be_error).cpu()  
        self.memory.add(weighted_error, (state, action, reward, next_state, done))
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
        dones = np.array([ss[4] for ss in mini_batch])

        dones = dones.astype(int)

        states = torch.Tensor(states).float().to(device)
        # states = Variable(states).float()
        pred = self.model(states)

        a = torch.LongTensor(actions).view(-1, 1).to(device)

        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_().to(device)
        one_hot_action.scatter_(1, a, 1)

        pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)

        next_states = torch.Tensor(next_states).float().to(device)
        # next_states = Variable(next_states).float()
        next_pred = self.target_model(next_states).data

        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        target = rewards + (1 - dones) * self.discount_factor * next_pred.max(1)[0]
        target = Variable(target)

        errors = torch.abs(pred - target).cpu().data.numpy()

        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        is_weights_tensor = torch.FloatTensor(is_weights).to(device)
        loss = (is_weights_tensor * F.mse_loss(pred, target)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
        print(f'######Round {j+1}######')
        # env = gym.make('CartPole-v0')
        # env = gym.make('CartPole-v1', new_step_api=True)
        # env = gym.make('MountainCar-v0', new_step_api=True)
        # env = gym.make('Acrobot-v1')
        env = Env1(j)
        state_size = env_params['state_dim']
        action_size = env_params['action_dim']
        max_steps = env_params['max_steps']
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        print(f"Using device: {device}")
        model = DQN(state_size, action_size, device)
        agent = DQNAgent(state_size, action_size, device)

        best_steps = np.inf
        return_list = []
        sequence = {}
        best_model = j
        """
        Initialization of gradient-based weight-adjustment
        """
        agent.w1 = 0.2
        agent.w2 = 0.8
        zeta = 2.2
        lr = 0.001

        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    done = False
                    step = 0
                    # ep_reward = 0
                    arrive_num = 0
                    episode_return = 0
                    agent_sequence = np.array(env_params['agent_start']).reshape(1, 2)
                    state = env.reset()  # 42
                    state = np.reshape(state, [1, state_size])

                    """
                    Gradient-based optimization of designated weights to the TD and BE errors
                    """
                    dw1 = 2 * (zeta - (agent.w1 / agent.w2)) * (-1 / (agent.w1 + agent.w2) ** 2)
                    dw2 = -dw1
                    agent.w1 -= dw1 * lr
                    agent.w2 -= dw2 * lr

                    while not done:

                        if done:
                            break
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

                    if info['arrive'] == 1:
                        if len(agent_sequence) < best_steps:
                            best_steps = len(agent_sequence)
                            agent.save_model(best_model)

                    agent.update_target_model()

                    sequence['episode' + str(num_episodes / 10 * i + i_episode + 1)] = agent_sequence
                    return_list.append(episode_return)
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode':
                                '%d' % (num_episodes / 10 * i + i_episode + 1),
                            'return':
                                '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)




        path2 = f'{save_path}/return_List_env' + f'{j}' + '.txt'
        np.savetxt(path2, np.array(return_list))  

        path1 = f'{save_path}/agent_sequence_env' + f'{j}' + '.npy'
        np.save(path1, sequence)  

