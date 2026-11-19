import sys
import gym
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from SORLPER.prioritized_memory import Memory
from batch_Parameters import env_params, agent_params
from batch_env1 import Env1

""" Training Parameters """
num_episodes = env_params['num_trials']
DISCOUNT = agent_params['discount_rate']
LEARNING_RATE = agent_params['learning_rate']
MEMORY_SIZE = agent_params['buffer_size']
EPSILON_START = agent_params['start_epsilon']
EPSILON_END = agent_params['final_epsilon']
BATCH_SIZE = agent_params['batch_size']
TRAIN_START = agent_params['minimal_size']
W1 = agent_params['w1']
W2 = agent_params['w2']



class DeepSOM(object):

    def __init__(self, input_dim, map_size, learning_rate, sigma, sigma_const):

        self.SOM_layer = SOMLayer(input_dim, map_size, learning_rate, sigma, sigma_const)

        return

    def Update(self, state, best_unit, reward_value):

        self.SOM_layer.Update(state, best_unit, reward_value)

        return

    def GetOutput(self, state):

        best_unit = self.SOM_layer.GetBestUnit(state)

        return best_unit


class SOMLayer():

    def __init__(self, input_dim, size, learning_rate, sigma, sigma_const):

        self.size = size
        self.num_units = size * size
        self.num_weights = input_dim

        self.learning_rate = learning_rate
        self.sigma = sigma
        self.sigma_const = sigma_const

        self.units = {'xy': [], 'w': []}
        self.ConstructMap()

        return

    def ConstructMap(self):

        x = 0
        y = 0

        # Construct map
        for u in range(self.num_units):

            self.units['xy'].append([x, y])
            self.units['w'].append(np.random.rand(self.num_weights))#np.random.randn(self.num_weights))

            x += 1

            if (x >= self.size):
                x = 0
                y += 1

        self.units['xy'] = np.array(self.units['xy'])
        self.units['w'] = np.array(self.units['w'])

        return

    def Update(self, state, best_unit, reward_value):
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        diffs = self.units['xy'] - self.units['xy'][best_unit, :]
        location_distances = np.sqrt(np.sum(np.square(diffs), axis=-1))
        neighbourhood_values = np.exp(-np.square(location_distances) / (2.0 * (self.sigma_const + (reward_value * self.sigma))))

        self.units['w'] += (reward_value * self.learning_rate) * \
                           np.expand_dims(neighbourhood_values, axis=-1) * (state - self.units['w'])

        return

    def GetBestUnit(self, state):

        best_unit = np.argmin(np.sum((self.units['w'] - state) ** 2, axis=-1), axis=0)

        return best_unit


""" Feedforward network as the Q function """
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
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
    def __init__(self, state_size, action_size, env_params, agent_params):

        self.render = False
        self.load_model = False
        self.bSOM = agent_params['bSOM']

        self.weighting_decay = agent_params['w_decay']
        self.TD_decay = agent_params['TD_decay']

        """ get the size of state and action spaces """
        self.state_size = state_size
        self.action_size = action_size

        """ training hyperparameters """
        self.discount_factor = DISCOUNT
        self.learning_rate = LEARNING_RATE
        self.memory_size = MEMORY_SIZE
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / num_episodes * 0.8   #self.explore_step
        self.batch_size = BATCH_SIZE
        self.train_start = TRAIN_START

        """ W1 is assigned to the TD error and W2 to the BE """
        self.w1 = W1
        self.w2 = W2
        self.td_buffer = deque(maxlen=self.memory_size)
        self.be_buffer = deque(maxlen=self.memory_size)

        self.memory = Memory(self.memory_size)
        self.CreateSOM(agent_params)

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.update_target_model()

    def CreateSOM(self, agent_params):

        self.SOM = DeepSOM(self.state_size, agent_params['SOM_size'],
                           agent_params['SOM_alpha'], agent_params['SOM_sigma'],
                           agent_params['SOM_sigma_const'])
        self.Q_alpha = agent_params['Q_alpha']
        self.QValues = np.zeros((agent_params['SOM_size'] * agent_params['SOM_size'], self.action_size))


    def update_target_model(self):
        """ to periodically update the target model """
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        """ the agent follows an epsilon-greedy exploration """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_temp = torch.from_numpy(state).float()
            with torch.no_grad():
                q_graph_values = self.model(state_temp).cpu().detach().numpy()
            q_values = self.GetQValues(state, q_graph_values)
            action = np.argmax(q_values)
            return int(action)

    def save_model(self, num):
        # Load the model of SOM
        torch.save(self.model.state_dict(), f"{env_params['save_path']}/Q_net{num}.pth")
        # Load the model of SOM
        file_path = f'C:/Users/37409/Desktop/SORLPER_Nav/SORLPER/' + f'{save_path}' + f'/data_QValues{num}.pkl'
        file_path2 = f'C:/Users/37409/Desktop/SORLPER_Nav/SORLPER/' + f'{save_path}' + f'/data_units{num}.pkl'

        with open(file_path, "wb") as f:
            pickle.dump(agent.QValues, f)

        with open(file_path2, "wb") as f:
            pickle.dump(agent.SOM.SOM_layer.units, f)

    def QGraphGetActionValues(self, state):
        state = torch.from_numpy(state).float()
        logits = self.model(state)
        # return q_value.numpy()
        return logits.cpu().detach().numpy()

    def QTargetGraphGetActionValues(self, state):
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            preds = self.target_model(state)
        return preds.cpu().detach().numpy()

    def append_sample(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state)
        current_qs = self.model(state_tensor).data
        q_target_graph_values = np.squeeze(np.array(self.QTargetGraphGetActionValues(np.expand_dims(next_state, axis=0))))

        if (self.bSOM):
            future_qs = self.GetQValues(next_state, q_target_graph_values)

        else:
            future_qs = q_target_graph_values

        future_qs = torch.from_numpy(future_qs)
        if not done:
            max_future_q = torch.max(future_qs)
            new_q = reward + self.discount_factor * max_future_q
        else:
            new_q = reward

        """ here we construct the prioritization score to be used for transitions rankings """
        td_error = torch.abs(new_q - current_qs[0][action])
        be_error = torch.abs(new_q - current_qs)
        mean_be_error = torch.mean(be_error)
        weighted_error = (self.w1 * td_error + self.w2 * mean_be_error).cpu()

        self.memory.add(weighted_error, (state, action, reward, next_state, done))

    def GetWeighting(self, best_unit, state):

        diff = np.sum(np.square(self.SOM.SOM_layer.units['w'][best_unit, :] - state))
        w = np.exp(-diff / self.weighting_decay)

        return w

    def GetQValues(self, state, q_graph_values):

        best_unit = self.SOM.GetOutput(state)
        som_action_values = self.QValues[best_unit, :]
        w = self.GetWeighting(best_unit, state)
        q_values = (w * som_action_values) + ((1 - w) * q_graph_values)

        return q_values

    def UpdateBatchSOM(self, states, actions, rewards, batch_targets, batch_bTrial_over):
        # Loop over each state in the batch
        self.BatchQValues = []
        states_np = states.cpu().numpy() if isinstance(states, torch.Tensor) else states
        for state, action, reward, target, bTrial_over in \
                zip(states_np, actions, rewards, batch_targets, batch_bTrial_over):
            target = self.GetTargetValue(bTrial_over, reward, state)

            best_unit = self.SOM.GetOutput(state)
            delta = np.exp(np.abs(target -
                                  np.squeeze(self.QGraphGetActionValues(
                                      np.expand_dims(state, axis=0)))[action]) / self.TD_decay) - 1

            delta = np.clip(delta, 0, 1)
            self.SOM.Update(state, best_unit, delta)

            best_unit = self.SOM.GetOutput(state)
            w = self.GetWeighting(best_unit, state)
            self.QValues[best_unit, action] += self.Q_alpha * w * (target - self.QValues[best_unit, action])
            self.BatchQValues.append(self.QValues[best_unit, action])

        return

    def GetTargetValue(self, bTrial_over, reward, state):

        q_graph_values = np.squeeze(np.array(self.QTargetGraphGetActionValues(np.expand_dims(state, axis=0))))

        if(self.bSOM):
            q_values = self.GetQValues(state, q_graph_values)

        else:
            q_values = q_graph_values

        max_q_value = np.amax(q_values)

        if (bTrial_over):
            target = reward

        else:
            target = reward + (max_q_value * self.discount_factor)

        return target

    def GetBatchTargetValues(self, batch_bTrial_over, rewards, states):
        targets = []

        # Loop over each state in the batch
        for bTrial_over, state, reward in zip(batch_bTrial_over, states, rewards):
            target = self.GetTargetValue(bTrial_over, reward, state)
            targets.append(target)

        return np.array(targets)

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
        batch_bTrial_over = dones

        batch_targets = self.GetBatchTargetValues(batch_bTrial_over, rewards, next_states)

        if (self.bSOM):
            self.UpdateBatchSOM(states, actions, rewards, batch_targets, batch_bTrial_over)
            # self.q_graph.GradientDescentStep(mini_batch, np.array(batch_targets), idxs, is_weights, self.memory)

        dones = dones.astype(int)

        states = torch.Tensor(states).float()
        pred = self.model(states)

        a = torch.LongTensor(actions).view(-1, 1)

        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)

        pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)

        batch_targets = torch.FloatTensor(batch_targets)
        target = Variable(batch_targets)

        errors = torch.abs(pred - target).cpu().data.numpy()

        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        is_weights_tensor = torch.FloatTensor(is_weights)
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
        print(f'######Round {j + 1}######')
        # env = gym.make('CartPole-v0')
        # env = gym.make('CartPole-v1', new_step_api=True)
        # env = gym.make('MountainCar-v0', new_step_api=True)
        # env = gym.make('Acrobot-v1')
        env = Env1(j)
        state_size = env_params['state_dim']
        action_size = env_params['action_dim']
        max_steps = env_params['max_steps']
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {device}")
        model = DQN(state_size, action_size)
        agent = DQNAgent(state_size, action_size, env_params, agent_params)

        best_steps = np.inf
        return_list = []
        sequence = {}
        best_model = j
        """
        Initialization of gradient-based weight-adjustment
        """
        agent.w1 = agent_params['w1']
        agent.w2 = agent_params['w2']
        zeta = agent_params['zeta']
        lr = agent_params['w1_w2_lr']

        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    done = False
                    step = 0
                    arrive_num = 0
                    episode_return = 0
                    agent_sequence = np.array(env_params['agent_start']).reshape(1, 2)
                    state = env.reset()  # 42
                    state = np.reshape(state, [1, state_size])

                    """
                    Gradient-based optimization of designated weights to the TD and BE errors
                    """
                    if agent_params['dynamic_w1_w2']:
                        dw1 = 2 * (zeta - (agent.w1 / agent.w2)) * (-1 / (agent.w1 + agent.w2) ** 2)
                        dw2 = -dw1
                        agent.w1 -= dw1 * lr
                        agent.w2 -= dw2 * lr

                    else:
                        agent.w1 = agent_params['w1']
                        agent.w2 = agent_params['w2']

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
        # Save the rewards and movement paths
        path2 = f'{save_path}/return_List_env' + f'{j}' + '.txt'
        np.savetxt(path2, np.array(return_list))

        path1 = f'{save_path}/agent_sequence_env' + f'{j}' + '.npy'
        np.save(path1, sequence)
