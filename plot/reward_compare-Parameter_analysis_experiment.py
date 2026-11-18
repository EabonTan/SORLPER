import numpy as np
import seaborn as sns
import warnings
from batch_Parameters import env_params, agent_params
# Add the following configuration immediately after importing matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

# Configure Matplotlib to use the LaTeX engine for rendering mathematical symbols
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Add the amsmath package to support Greek letters

# If the above method does not work, try the following alternatives
# mpl.rcParams['mathtext.fontset'] = 'stix'  # Use the STIX font set
# mpl.rcParams['font.family'] = 'STIXGeneral'  # Set STIX as the default font
# Ignore specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Define the improved moving average function
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# Experimental parameters
num_episodes = env_params['num_trials']
num_runs = 5

# Store reward data for different parameter settings
all_rewards_params = [[] for _ in range(7)]  # 7 parameter settings

# Labels corresponding to the parameter settings (named according to the parameter values in the paths)
param_labels = [
    r'$\kappa_1$=0.01,$\kappa_2$=0.99',
    r'$\kappa_1$=0.1,$\kappa_2$=0.9',
    r'$\kappa_1$=0.2,$\kappa_2$=0.8',
    r'$\kappa_1$=0.5,$\kappa_2$=0.5',
    r'$\kappa_1$=0.8,$\kappa_2$=0.2',
    r'$\kappa_1$=0.9,$\kappa_2$=0.1',
    r'$\kappa_1$=0.99,$\kappa_2$=0.01'
]

# Color scheme (ensure that there are 7 distinct colors)
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']

# Load reward data for each run
for i in range(num_runs):
    # Define the paths for the 7 different parameter settings
    file_paths = [
        f'E:/FILE/SORLPER_Nav/SORLPER/results_ParaAna - 0.0199/return_List_env{i}.txt',
        f'E:/FILE/SORLPER_Nav/SORLPER/results_ParaAna - 19/return_List_env{i}.txt',
        f'E:/FILE/SORLPER_Nav/SORLPER/results_ParaAna - 28/return_List_env{i}.txt',
        f'E:/FILE/SORLPER_Nav/SORLPER/results_ParaAna - 0.5/return_List_env{i}.txt',
        f'E:/FILE/SORLPER_Nav/SORLPER/results_ParaAna - 82/return_List_env{i}.txt',
        f'E:/FILE/SORLPER_Nav/SORLPER/results_ParaAna - 91/return_List_env{i}.txt',
        f'E:/FILE/SORLPER_Nav/SORLPER/results_ParaAna - 0.991/return_List_env{i}.txt'
    ]

    # Load data for each parameter setting
    for j in range(7):
        try:
            rewards = np.loadtxt(file_paths[j])
            all_rewards_params[j].append(rewards)
        except Exception as e:
            print(f"Error loading file: {file_paths[j]}")
            print(f"Error message: {str(e)}")
            # Create an empty array as a placeholder
            all_rewards_params[j].append(np.zeros(num_episodes))

# Convert the lists into NumPy arrays
all_rewards_params = [np.array(rewards) for rewards in all_rewards_params]

# Apply moving average smoothing to each reward curve (window size set to 9)
window_size = 9
smoothed_rewards_params = []
for rewards in all_rewards_params:
    smoothed = []
    for run in rewards:
        # Apply the improved moving average function
        smoothed_run = moving_average(run, window_size=window_size)
        smoothed.append(smoothed_run)
    smoothed_rewards_params.append(np.array(smoothed))

# Calculate the mean and standard deviation of the smoothed rewards for each parameter setting
smoothed_means = []
smoothed_stds = []

for rewards in smoothed_rewards_params:
    # Calculate the mean along the time axis, ignoring NaN values
    mean_vals = []
    std_vals = []

    # Iterate over each time step
    for t in range(rewards.shape[1]):
        # Obtain the data from all runs at the current time step
        vals = rewards[:, t]
        # Consider only non-NaN values
        valid_vals = vals[~np.isnan(vals)]

        if len(valid_vals) > 0:
            mean_vals.append(np.mean(valid_vals))
            std_vals.append(np.std(valid_vals))
        else:
            # Use NaN if there are no valid values
            mean_vals.append(np.nan)
            std_vals.append(np.nan)

    smoothed_means.append(np.array(mean_vals))
    smoothed_stds.append(np.array(std_vals))

# Calculate the smoothed upper and lower bounds (mean ± standard deviation)
smoothed_upper_bounds = [mean + std for mean, std in zip(smoothed_means, smoothed_stds)]
smoothed_lower_bounds = [mean - std for mean, std in zip(smoothed_means, smoothed_stds)]

# Generate the x-axis coordinates (using the length of the first parameter setting)
smoothed_episodes_list = list(range(len(smoothed_means[0])))

# Plot the smoothed mean curves and their variation ranges for each parameter setting
plt.figure(figsize=(7, 5))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '12'

# Plot the 7 curves
for i in range(7):
    # Plot only valid data points (non-NaN)
    valid_indices = ~np.isnan(smoothed_means[i])
    x_values = np.array(smoothed_episodes_list)[valid_indices]
    y_mean = smoothed_means[i][valid_indices]
    y_upper = smoothed_upper_bounds[i][valid_indices]
    y_lower = smoothed_lower_bounds[i][valid_indices]

    plt.plot(x_values, y_mean, label=param_labels[i], color=colors[i])
    plt.fill_between(
        x_values,
        y_lower,
        y_upper,
        color=colors[i],
        alpha=0.2
    )

plt.xlabel('Number of experiments', font={'family': 'Times New Roman', 'size': 12})
plt.ylabel('Average Reward', font={'family': 'Times New Roman', 'size': 12})
# plt.title('Performance of SORLPER with Different $\kappa_1$ Values', fontsize=14)
plt.legend(loc="best", fontsize=10)  # Automatically select the best position for the legend
# plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the figures if needed
plt.savefig('Param_Analysis.pdf', dpi=1200)
plt.savefig('Param_Analysis.jpg', dpi=1200)
plt.savefig('Param_Analysis.eps', dpi=1200)
# plt.show()

# Calculate the average reward for each parameter setting over 10 experiments
print("\nAverage Reward for Each Parameter Setting:")
for i in range(7):
    # Calculate the mean of the original rewards (across episodes)
    mean_rewards = np.mean(all_rewards_params[i], axis=0)
    # Calculate the average reward over the entire training process (ignoring NaN values)
    valid_rewards = mean_rewards[~np.isnan(mean_rewards)]
    if len(valid_rewards) > 0:
        avg_reward = np.mean(valid_rewards)
    else:
        avg_reward = np.nan

    print(f"{param_labels[i]} Average Reward: {avg_reward:.3f}")
