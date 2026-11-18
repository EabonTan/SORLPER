import numpy as np
import matplotlib.pyplot as plt
# import rl_utils
import seaborn as sns
from batch_Parameters import env_params, agent_params

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# Experimental parameters
num_episodes = env_params['num_trials']
num_runs = env_params['num_repeats']

# Store all reward data for different algorithms
all_rewards_TD3 = []
all_rewards_DDPG = []
all_rewards_PER = []
all_rewards_DDQNPER = []
all_rewards_BETDQN = []
all_rewards_SORLPER = []
all_rewards_DSAC = []

save_path = 'results'      # env_params['save_path'] 

# Load reward data for each run
for i in range(num_runs):
    
    # Environment 1
    # file_path_TD3 = f'E:/FILE/SORLPER_Nav/TD3PER/' + f'{save_path}' + f'/return_List_env{i}' + '.txt'
    # file_path_DDPG = f'E:/FILE/SORLPER_Nav/DDPGPER/' + f'{save_path}' + f'/return_List_env{i}'+'.txt'
    # file_path_PER = f'E:/FILE/SORLPER_Nav/PER/' + f'{save_path}' + f'/return_List_env{i}'+'.txt'
    # file_path_DDQNPER = f'E:/FILE/SORLPER_Nav/DDQNPER/' + f'{save_path}' + f'/return_List_env{i}'+'.txt'
    # file_path_BETDQN = f'E:/FILE/SORLPER_Nav/BETDQN/' + f'{save_path}' + f'/return_List_env{i}'+'.txt'
    # file_path_SORLPER = f'E:/FILE/SORLPER_Nav/SORLPER/' + f'{save_path}' + f'/return_List_env{i}'+'.txt'
    # file_path_DSAC = f'E:/FILE/SORLPER_Nav/DSAC/' + f'{save_path}' + f'/return_List_env{i}'+'.txt'

    # Environment 2
    # file_path_TD3 = f'E:/FILE/SORLPER_Nav/TD3PER/' + f'{save_path}' + f'/return_List_env{i}' + '.txt'
    # file_path_DDPG = f'E:/FILE/SORLPER_Nav/DDPGPER/' + f'{save_path}' + f'/return_List_env{i}'+'.txt'
    # file_path_PER = f'E:/FILE/SORLPER_Nav/PER/' + f'{save_path}' + f'/return_List_env{i}'+'.txt'
    # file_path_DDQNPER = f'E:/FILE/SORLPER_Nav/DDQNPER/' + f'{save_path}' + f'/return_List_env{i}'+'.txt'
    # file_path_BETDQN = f'E:/FILE/SORLPER_Nav/BETDQN/' + f'{save_path}' + f'/return_List_env{i}'+'.txt'
    # file_path_SORLPER = f'E:/FILE/SORLPER_Nav/SORLPER/' + f'{save_path}' + f'/return_List_env{i}'+'.txt'
    # file_path_DSAC = f'E:/FILE/SORLPER_Nav/DSAC/' + f'{save_path}' + f'/return_List_env{i}'+'.txt'

    rewards_TD3 = np.loadtxt(file_path_TD3)
    all_rewards_TD3.append(rewards_TD3)
  
    rewards_DDPG = np.loadtxt(file_path_DDPG)
    all_rewards_DDPG.append(rewards_DDPG)
    
    rewards_PER = np.loadtxt(file_path_PER)
    all_rewards_PER.append(rewards_PER)

    rewards_DDQNPER = np.loadtxt(file_path_DDQNPER)
    all_rewards_DDQNPER.append(rewards_DDQNPER)

    rewards_BETDQN = np.loadtxt(file_path_BETDQN)
    all_rewards_BETDQN.append(rewards_BETDQN)

    rewards_SORLPER = np.loadtxt(file_path_SORLPER)
    all_rewards_SORLPER.append(rewards_SORLPER)

    rewards_DSAC = np.loadtxt(file_path_DSAC)
    all_rewards_DSAC.append(rewards_DSAC)

# Convert the lists into NumPy arrays (each with the shape (num_runs, num_episodes))
all_rewards_TD3 = np.array(all_rewards_TD3)
all_rewards_DDPG = np.array(all_rewards_DDPG)
all_rewards_PER = np.array(all_rewards_PER)
all_rewards_DDQNPER = np.array(all_rewards_DDQNPER)
all_rewards_BETDQN = np.array(all_rewards_BETDQN)
all_rewards_SORLPER = np.array(all_rewards_SORLPER)
all_rewards_DSAC = np.array(all_rewards_DSAC)

# Optionally, calculate statistics for the original reward data
mean_rewards_TD3 = np.mean(all_rewards_TD3, axis=0)
std_rewards_TD3 = np.std(all_rewards_TD3, axis=0)
mean_rewards_DDPG = np.mean(all_rewards_DDPG, axis=0)
std_rewards_DDPG = np.std(all_rewards_DDPG, axis=0)
mean_rewards_PER = np.mean(all_rewards_PER, axis=0)
std_rewards_PER = np.std(all_rewards_PER, axis=0)
mean_rewards_DDQNPER = np.mean(all_rewards_DDQNPER, axis=0)
std_rewards_DDQNPER = np.std(all_rewards_DDQNPER, axis=0)
mean_rewards_BETDQN = np.mean(all_rewards_BETDQN, axis=0)
std_rewards_BETDQN = np.std(all_rewards_BETDQN, axis=0)
mean_rewards_SORLPER = np.mean(all_rewards_SORLPER, axis=0)
std_rewards_SORLPER = np.std(all_rewards_SORLPER, axis=0)
mean_rewards_DSAC = np.mean(all_rewards_DSAC, axis=0)
std_rewards_DSAC = np.std(all_rewards_DSAC, axis=0)

# Apply moving average smoothing to each reward curve (window size set to 9)
smoothed_rewards_TD3 = np.apply_along_axis(moving_average, axis=1, arr=all_rewards_TD3, window_size=9)
smoothed_rewards_DDPG = np.apply_along_axis(moving_average, axis=1, arr=all_rewards_DDPG, window_size=9)
smoothed_rewards_PER = np.apply_along_axis(moving_average, axis=1, arr=all_rewards_PER, window_size=9)
smoothed_rewards_DDQNPER = np.apply_along_axis(moving_average, axis=1, arr=all_rewards_DDQNPER, window_size=9)
smoothed_rewards_BETDQN = np.apply_along_axis(moving_average, axis=1, arr=all_rewards_BETDQN, window_size=9)
smoothed_rewards_SORLPER = np.apply_along_axis(moving_average, axis=1, arr=all_rewards_SORLPER, window_size=9)
smoothed_rewards_DSAC = np.apply_along_axis(moving_average, axis=1, arr=all_rewards_DSAC, window_size=9)

# Calculate the mean and standard deviation of the smoothed rewards
smoothed_mean_rewards_TD3 = np.mean(smoothed_rewards_TD3, axis=0)
smoothed_std_rewards_TD3 = np.std(smoothed_rewards_TD3, axis=0)

smoothed_mean_rewards_DDPG = np.mean(smoothed_rewards_DDPG, axis=0)
smoothed_std_rewards_DDPG = np.std(smoothed_rewards_DDPG, axis=0)

smoothed_mean_rewards_PER = np.mean(smoothed_rewards_PER, axis=0)
smoothed_std_rewards_PER = np.std(smoothed_rewards_PER, axis=0)

smoothed_mean_rewards_DDQNPER = np.mean(smoothed_rewards_DDQNPER, axis=0)
smoothed_std_rewards_DDQNPER = np.std(smoothed_rewards_DDQNPER, axis=0)

smoothed_mean_rewards_BETDQN = np.mean(smoothed_rewards_BETDQN, axis=0)
smoothed_std_rewards_BETDQN = np.std(smoothed_rewards_BETDQN, axis=0)

smoothed_mean_rewards_SORLPER = np.mean(smoothed_rewards_SORLPER, axis=0)
smoothed_std_rewards_SORLPER = np.std(smoothed_rewards_SORLPER, axis=0)

smoothed_mean_rewards_DSAC = np.mean(smoothed_rewards_DSAC, axis=0)
smoothed_std_rewards_DSAC = np.std(smoothed_rewards_DSAC, axis=0)

# Calculate the smoothed upper and lower bounds (mean ± standard deviation)
smoothed_upper_bound_TD3 = smoothed_mean_rewards_TD3 + smoothed_std_rewards_TD3
smoothed_lower_bound_TD3 = smoothed_mean_rewards_TD3 - smoothed_std_rewards_TD3

smoothed_upper_bound_DDPG = smoothed_mean_rewards_DDPG + smoothed_std_rewards_DDPG
smoothed_lower_bound_DDPG = smoothed_mean_rewards_DDPG - smoothed_std_rewards_DDPG

smoothed_upper_bound_PER = smoothed_mean_rewards_PER + smoothed_std_rewards_PER
smoothed_lower_bound_PER = smoothed_mean_rewards_PER - smoothed_std_rewards_PER

smoothed_upper_bound_DDQNPER = smoothed_mean_rewards_DDQNPER + smoothed_std_rewards_DDQNPER
smoothed_lower_bound_DDQNPER = smoothed_mean_rewards_DDQNPER - smoothed_std_rewards_DDQNPER

smoothed_upper_bound_BETDQN = smoothed_mean_rewards_BETDQN + smoothed_std_rewards_BETDQN
smoothed_lower_bound_BETDQN = smoothed_mean_rewards_BETDQN - smoothed_std_rewards_BETDQN

smoothed_upper_bound_SORLPER = smoothed_mean_rewards_SORLPER + smoothed_std_rewards_SORLPER
smoothed_lower_bound_SORLPER = smoothed_mean_rewards_SORLPER - smoothed_std_rewards_SORLPER

smoothed_upper_bound_DSAC = smoothed_mean_rewards_DSAC + smoothed_std_rewards_DSAC
smoothed_lower_bound_DSAC = smoothed_mean_rewards_DSAC - smoothed_std_rewards_DSAC

# Generate the x-axis coordinates (the length of the smoothed reward curves may vary slightly)
smoothed_episodes_list = list(range(len(smoothed_mean_rewards_SORLPER)))

# Plot the smoothed mean curves and their variation ranges for each algorithm
plt.figure(figsize=(7, 5))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '12'

plt.plot(smoothed_episodes_list, smoothed_mean_rewards_PER, label=f'PER({np.mean(mean_rewards_PER):.3f})', color='purple')
plt.fill_between(smoothed_episodes_list, smoothed_lower_bound_PER, smoothed_upper_bound_PER, color='purple', alpha=0.2)

plt.plot(smoothed_episodes_list, smoothed_mean_rewards_DDPG, label=f'DDPGPER({np.mean(mean_rewards_DDPG):.3f})', color='red')
plt.fill_between(smoothed_episodes_list, smoothed_lower_bound_DDPG, smoothed_upper_bound_DDPG, color='red', alpha=0.2)

plt.plot(smoothed_episodes_list, smoothed_mean_rewards_TD3, label=f'TD3PER({np.mean(mean_rewards_TD3):.3f})')
plt.fill_between(smoothed_episodes_list, smoothed_lower_bound_TD3, smoothed_upper_bound_TD3, alpha=0.2)

plt.plot(smoothed_episodes_list, smoothed_mean_rewards_BETDQN, label=f'BETDQN({np.mean(mean_rewards_BETDQN):.3f})', color='green')
plt.fill_between(smoothed_episodes_list, smoothed_lower_bound_BETDQN, smoothed_upper_bound_BETDQN, color='green', alpha=0.2)

plt.plot(smoothed_episodes_list, smoothed_mean_rewards_DSAC, label=f'DSACPER({np.mean(mean_rewards_DSAC):.3f})', color='brown')
plt.fill_between(smoothed_episodes_list, smoothed_lower_bound_DSAC, smoothed_upper_bound_DSAC, color='brown', alpha=0.2)

plt.plot(smoothed_episodes_list, smoothed_mean_rewards_DDQNPER, label=f'DDQNPER({np.mean(mean_rewards_DDQNPER):.3f})', color='orange')
plt.fill_between(smoothed_episodes_list, smoothed_lower_bound_DDQNPER, smoothed_upper_bound_DDQNPER, color='orange', alpha=0.2)

plt.plot(smoothed_episodes_list, smoothed_mean_rewards_SORLPER, label=f'SORLPER({np.mean(mean_rewards_SORLPER):.3f})', color='blue')
plt.fill_between(smoothed_episodes_list, smoothed_lower_bound_SORLPER, smoothed_upper_bound_SORLPER, color='blue', alpha=0.2)


plt.xlabel('Number of experiments', font={'family': 'Times New Roman', 'size': 12})
plt.ylabel('Average Reward', font={'family': 'Times New Roman', 'size': 12})
# plt.legend(loc="lower right")
# plt.grid(True)
# Place the legend above the plot, arranged in 3 or 4 columns
plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=3, frameon=True)
plt.tight_layout()

# Alternatively, place the legend inside the upper-left corner of the plot, arranged in 2 columns
# plt.legend(loc="upper left", ncol=2)
# To save the figure, uncomment the following lines and adjust the save path
# plt.savefig('E:/FILE/SORLPER_Nav/Fig9a.jpg', dpi=1200)
# plt.savefig('E:/FILE/SORLPER_Nav/Fig9a.pdf', dpi=1200)
plt.show()

# Calculate the average reward over 500 episodes across 10 experiments for each algorithm
avg_SORLPER = np.mean(mean_rewards_SORLPER)
avg_BETDQN = np.mean(mean_rewards_BETDQN)
avg_PER = np.mean(mean_rewards_PER)
avg_DDQNPER = np.mean(mean_rewards_DDQNPER)
avg_DDPG = np.mean(mean_rewards_DDPG)
avg_TD3 = np.mean(mean_rewards_TD3)
avg_DSAC = np.mean(mean_rewards_DSAC)
print(f"SORLPER Average Reward: {avg_SORLPER:.3f}")
print(f"BETDQN Average Reward: {avg_BETDQN:.3f}")
print(f"PER Average Reward: {avg_PER:.3f}")
print(f"DDQNPER Average Reward: {avg_DDQNPER:.3f}")
print(f"DDPG Average Reward: {avg_DDPG:.3f}")
print(f"TD3 Average Reward: {avg_TD3:.3f}")
print(f"DSAC Average Reward: {avg_DSAC:.3f}")
