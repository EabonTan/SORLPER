import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from batch_Parameters import env_params, agent_params


# =============================================================================
# 1. Experimental parameters
# =============================================================================

num_episodes = env_params['num_trials']
num_runs = env_params['num_repeats']

save_path = 'results-2'


# =============================================================================
# 2. Store reward data
# =============================================================================

all_rewards_RLPER = []
all_rewards_SORL = []
all_rewards_SORLPER_TD = []
all_rewards_SORLPER = []


# =============================================================================
# 3. Load reward data from different algorithms
# =============================================================================

for i in range(num_runs):

    # -------------------------------------------------------------------------
    # RLPER
    # -------------------------------------------------------------------------
    file_path_RLPER = (
        f'E:/FILE/SORLPER_Nav/RLPER/{save_path}/return_List_env{i}.txt'
    )

    rewards_RLPER = np.loadtxt(file_path_RLPER)
    all_rewards_RLPER.append(rewards_RLPER)


    # -------------------------------------------------------------------------
    # SORL
    # -------------------------------------------------------------------------
    file_path_SORL = (
        f'E:/FILE/SORLPER_Nav/SORL/{save_path}/return_List_env{i}.txt'
    )

    rewards_SORL = np.loadtxt(file_path_SORL)
    all_rewards_SORL.append(rewards_SORL)


    # -------------------------------------------------------------------------
    # SORLPER-TD
    # -------------------------------------------------------------------------
    file_path_SORLPER_TD = (
        f'E:/FILE/SORLPER_Nav/SORLPER-TD/{save_path}/return_List_env{i}.txt'
    )

    rewards_SORLPER_TD = np.loadtxt(file_path_SORLPER_TD)
    all_rewards_SORLPER_TD.append(rewards_SORLPER_TD)


    # -------------------------------------------------------------------------
    # SORLPER
    # -------------------------------------------------------------------------
    file_path_SORLPER = (
        f'E:/FILE/SORLPER_Nav/SORLPER/{save_path}/return_List_env{i}.txt'
    )

    rewards_SORLPER = np.loadtxt(file_path_SORLPER)
    all_rewards_SORLPER.append(rewards_SORLPER)


# =============================================================================
# 4. Convert lists to NumPy arrays
# =============================================================================

all_rewards_RLPER = np.array(all_rewards_RLPER)
all_rewards_SORL = np.array(all_rewards_SORL)
all_rewards_SORLPER_TD = np.array(all_rewards_SORLPER_TD)
all_rewards_SORLPER = np.array(all_rewards_SORLPER)


# =============================================================================
# 5. Calculate mean and standard deviation
# =============================================================================

mean_rewards_RLPER = np.mean(all_rewards_RLPER, axis=0)
std_rewards_RLPER = np.std(all_rewards_RLPER, axis=0)

mean_rewards_SORL = np.mean(all_rewards_SORL, axis=0)
std_rewards_SORL = np.std(all_rewards_SORL, axis=0)

mean_rewards_SORLPER_TD = np.mean(all_rewards_SORLPER_TD, axis=0)
std_rewards_SORLPER_TD = np.std(all_rewards_SORLPER_TD, axis=0)

mean_rewards_SORLPER = np.mean(all_rewards_SORLPER, axis=0)
std_rewards_SORLPER = np.std(all_rewards_SORLPER, axis=0)


# =============================================================================
# 6. Moving average smoothing
# =============================================================================

window_size = 9

smoothed_rewards_RLPER = np.apply_along_axis(
    rl_utils.moving_average,
    axis=1,
    arr=all_rewards_RLPER,
    window_size=window_size
)

smoothed_rewards_SORL = np.apply_along_axis(
    rl_utils.moving_average,
    axis=1,
    arr=all_rewards_SORL,
    window_size=window_size
)

smoothed_rewards_SORLPER_TD = np.apply_along_axis(
    rl_utils.moving_average,
    axis=1,
    arr=all_rewards_SORLPER_TD,
    window_size=window_size
)

smoothed_rewards_SORLPER = np.apply_along_axis(
    rl_utils.moving_average,
    axis=1,
    arr=all_rewards_SORLPER,
    window_size=window_size
)


# =============================================================================
# 7. Calculate smoothed mean and standard deviation
# =============================================================================

smoothed_mean_rewards_RLPER = np.mean(
    smoothed_rewards_RLPER, axis=0
)
smoothed_std_rewards_RLPER = np.std(
    smoothed_rewards_RLPER, axis=0
)

smoothed_mean_rewards_SORL = np.mean(
    smoothed_rewards_SORL, axis=0
)
smoothed_std_rewards_SORL = np.std(
    smoothed_rewards_SORL, axis=0
)

smoothed_mean_rewards_SORLPER_TD = np.mean(
    smoothed_rewards_SORLPER_TD, axis=0
)
smoothed_std_rewards_SORLPER_TD = np.std(
    smoothed_rewards_SORLPER_TD, axis=0
)

smoothed_mean_rewards_SORLPER = np.mean(
    smoothed_rewards_SORLPER, axis=0
)
smoothed_std_rewards_SORLPER = np.std(
    smoothed_rewards_SORLPER, axis=0
)


# =============================================================================
# 8. Calculate upper and lower bounds
# =============================================================================

smoothed_upper_bound_RLPER = (
    smoothed_mean_rewards_RLPER +
    smoothed_std_rewards_RLPER
)

smoothed_lower_bound_RLPER = (
    smoothed_mean_rewards_RLPER -
    smoothed_std_rewards_RLPER
)


smoothed_upper_bound_SORL = (
    smoothed_mean_rewards_SORL +
    smoothed_std_rewards_SORL
)

smoothed_lower_bound_SORL = (
    smoothed_mean_rewards_SORL -
    smoothed_std_rewards_SORL
)


smoothed_upper_bound_SORLPER_TD = (
    smoothed_mean_rewards_SORLPER_TD +
    smoothed_std_rewards_SORLPER_TD
)

smoothed_lower_bound_SORLPER_TD = (
    smoothed_mean_rewards_SORLPER_TD -
    smoothed_std_rewards_SORLPER_TD
)


smoothed_upper_bound_SORLPER = (
    smoothed_mean_rewards_SORLPER +
    smoothed_std_rewards_SORLPER
)

smoothed_lower_bound_SORLPER = (
    smoothed_mean_rewards_SORLPER -
    smoothed_std_rewards_SORLPER
)


# =============================================================================
# 9. Generate x-axis
# =============================================================================

smoothed_episodes_list = np.arange(
    len(smoothed_mean_rewards_RLPER)
)


# =============================================================================
# 10. Plot
# =============================================================================

plt.figure(figsize=(7, 5))

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


# -----------------------------------------------------------------------------
# 1) RLPER
# -----------------------------------------------------------------------------

plt.plot(
    smoothed_episodes_list,
    smoothed_mean_rewards_RLPER,
    label='RLPER',
    color='red',
    linewidth=1.5
)

plt.fill_between(
    smoothed_episodes_list,
    smoothed_lower_bound_RLPER,
    smoothed_upper_bound_RLPER,
    color='red',
    alpha=0.2
)


# -----------------------------------------------------------------------------
# 2) SORL
# -----------------------------------------------------------------------------

plt.plot(
    smoothed_episodes_list,
    smoothed_mean_rewards_SORL,
    label='SORL',
    color='orange',
    linewidth=1.5
)

plt.fill_between(
    smoothed_episodes_list,
    smoothed_lower_bound_SORL,
    smoothed_upper_bound_SORL,
    color='orange',
    alpha=0.2
)


# -----------------------------------------------------------------------------
# 3) SORLPER-TD
# -----------------------------------------------------------------------------

plt.plot(
    smoothed_episodes_list,
    smoothed_mean_rewards_SORLPER_TD,
    label='SORLPER-TD',
    color='green',
    linewidth=1.5
)

plt.fill_between(
    smoothed_episodes_list,
    smoothed_lower_bound_SORLPER_TD,
    smoothed_upper_bound_SORLPER_TD,
    color='green',
    alpha=0.2
)


# -----------------------------------------------------------------------------
# 4) SORLPER
# -----------------------------------------------------------------------------

plt.plot(
    smoothed_episodes_list,
    smoothed_mean_rewards_SORLPER,
    label='SORLPER',
    color='blue',
    linewidth=1.5
)

plt.fill_between(
    smoothed_episodes_list,
    smoothed_lower_bound_SORLPER,
    smoothed_upper_bound_SORLPER,
    color='blue',
    alpha=0.2
)


# =============================================================================
# 11. Axis labels and legend
# =============================================================================

plt.xlabel(
    'Number of experiments',
    font={'family': 'Times New Roman', 'size': 12}
)

plt.ylabel(
    'Average Reward',
    font={'family': 'Times New Roman', 'size': 12}
)

plt.legend(loc='lower right')

# plt.grid(True)

plt.tight_layout()


# =============================================================================
# 12. Save figure
# =============================================================================

# plt.savefig(
#     'C:/Users/37409/Desktop/Figures/Fig6b.pdf',
#     dpi=1200,
#     bbox_inches='tight'
# )

# plt.savefig(
#     'C:/Users/37409/Desktop/Figures/Fig6b.jpg',
#     dpi=1200,
#     bbox_inches='tight'
# )

# plt.savefig(
#     'C:/Users/37409/Desktop/Figures/Fig6b.eps',
#     dpi=1200,
#     bbox_inches='tight'
# )


plt.show()


# =============================================================================
# 13. Calculate average reward over all episodes
# =============================================================================

avg_RLPER = np.mean(mean_rewards_RLPER)
avg_SORL = np.mean(mean_rewards_SORL)
avg_SORLPER_TD = np.mean(mean_rewards_SORLPER_TD)
avg_SORLPER = np.mean(mean_rewards_SORLPER)


print('============================================================')
print('Average Reward over all episodes')
print('============================================================')

print(f'RLPER       average reward: {avg_RLPER:.3f}')
print(f'SORL        average reward: {avg_SORL:.3f}')
print(f'SORLPER-TD  average reward: {avg_SORLPER_TD:.3f}')
print(f'SORLPER     average reward: {avg_SORLPER:.3f}')
