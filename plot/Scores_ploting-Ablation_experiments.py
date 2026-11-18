import numpy as np
import matplotlib.pyplot as plt
from batch_Parameters import env_params, agent_params


# =============================================================================
# 1. Experimental parameters
# =============================================================================

num_episodes = env_params['num_trials']       # e.g., 500
num_runs = env_params['num_repeats']          # e.g., 10

friend = np.array(env_params['agent_goal'])   # Target position

save_path = 'results-2'                          # Results folder name


# =============================================================================
# 2. File paths of the four algorithms
# =============================================================================

file_paths = {

    # -------------------------------------------------------------------------
    # RLPER
    # -------------------------------------------------------------------------
    'RLPER':
        f'E:/FILE/SORLPER_Nav/RLPER/{save_path}/agent_sequence_env{{}}.npy',

    # -------------------------------------------------------------------------
    # SORL
    # -------------------------------------------------------------------------
    'SORL':
        f'E:/FILE/SORLPER_Nav/SORL/{save_path}/agent_sequence_env{{}}.npy',

    # -------------------------------------------------------------------------
    # SORLPER-TD
    # -------------------------------------------------------------------------
    'SORLPER-TD':
        f'E:/FILE/SORLPER_Nav/SORLPER-TD/{save_path}/agent_sequence_env{{}}.npy',

    # -------------------------------------------------------------------------
    # SORLPER
    # -------------------------------------------------------------------------
    'SORLPER':
        f'E:/FILE/SORLPER_Nav/SORLPER/{save_path}/agent_sequence_env{{}}.npy'
}


# =============================================================================
# 3. Function for extracting episode records
# =============================================================================

def get_record(file_path, num_episodes, friend):
    """
    Load agent_sequence data from one experimental run.

    For each episode:
        - Column 0: number of steps
        - Column 1: success flag

    A trial is considered successful if the final position is
    within a distance of 10 from the target position.

    Returns
    -------
    record : ndarray
        Shape: (num_episodes, 2)
        Column 0: number of steps
        Column 1: success flag (1 = success, 0 = failure)
    """

    load_dict = np.load(
        file_path,
        allow_pickle=True
    ).item()

    record = np.zeros((num_episodes, 2))

    for ep in range(1, num_episodes + 1):

        key = "episode" + str(float(ep))

        if key in load_dict:

            episode = load_dict[key]

            # Number of steps
            steps = len(episode) - 1

            record[ep - 1, 0] = steps

            # -------------------------------------------------------------
            # Determine whether the robot reaches the target
            # -------------------------------------------------------------
            last_location = episode[-1, :]

            distance_to_goal = np.linalg.norm(
                last_location - friend
            )

            record[ep - 1, 1] = (
                1 if distance_to_goal < 10 else 0
            )

        else:

            record[ep - 1, :] = np.nan

    return record


# =============================================================================
# 4. Load all experimental results
# =============================================================================

cumulative_results = {}


for algo, path_template in file_paths.items():

    cumulative_success_list = []

    print('\n' + '=' * 70)
    print(f'Loading data for {algo}')
    print('=' * 70)

    for run in range(num_runs):

        file_path = path_template.format(run)

        try:

            record = get_record(
                file_path,
                num_episodes,
                friend
            )

            # -------------------------------------------------------------
            # Success flag of each episode
            # -------------------------------------------------------------
            success_flags = record[:, 1]

            # -------------------------------------------------------------
            # Cumulative number of successful episodes
            # -------------------------------------------------------------
            cumulative_success = np.cumsum(success_flags)

            cumulative_success_list.append(
                cumulative_success
            )

            print(
                f'{algo}: run {run} loaded successfully.'
            )

        except Exception as e:

            print(
                f'{algo}: run {run} data loading failed: {e}'
            )

    # -------------------------------------------------------------------------
    # Combine all runs
    # -------------------------------------------------------------------------

    if len(cumulative_success_list) > 0:

        cumulative_results[algo] = np.vstack(
            cumulative_success_list
        )

        print(
            f'{algo}: {len(cumulative_success_list)} runs loaded.'
        )

    else:

        print(
            f'{algo}: No valid data were loaded.'
        )


# =============================================================================
# 5. Calculate mean and standard deviation
# =============================================================================

mean_std_results = {}


for algo, arr in cumulative_results.items():

    mean_curve = np.mean(
        arr,
        axis=0
    )

    std_curve = np.std(
        arr,
        axis=0
    )

    mean_std_results[algo] = (
        mean_curve,
        std_curve
    )


# =============================================================================
# 6. Define plotting order
# =============================================================================

plot_order = [
    'RLPER',
    'SORL',
    'SORLPER-TD',
    'SORLPER'
]


print('\nFinal cumulative success:')
for algo in plot_order:
    if algo in cumulative_results:
        final_success = cumulative_results[algo][:, -1]
        print(
            f'{algo}: '
            f'{np.mean(final_success):.2f} ± {np.std(final_success):.2f}'
        )

# =============================================================================
# 7. Define colors
# =============================================================================

colors = {
    'RLPER': 'red',
    'SORL': 'orange',
    'SORLPER-TD': 'green',
    'SORLPER': 'blue'
}


# =============================================================================
# 8. Plot cumulative success curves
# =============================================================================

plt.figure(figsize=(7, 5))

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

episodes = np.arange(
    1,
    num_episodes + 1
)


for algo in plot_order:

    # Skip algorithm if no valid data were loaded
    if algo not in mean_std_results:
        print(
            f'Warning: {algo} is not available and will be skipped.'
        )
        continue

    mean_curve, std_curve = mean_std_results[algo]

    # -------------------------------------------------------------------------
    # Mean curve
    # -------------------------------------------------------------------------

    plt.plot(
        episodes,
        mean_curve,
        label=algo,
        color=colors[algo],
        linewidth=2
    )

    # -------------------------------------------------------------------------
    # Standard deviation region
    # -------------------------------------------------------------------------

    plt.fill_between(
        episodes,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color=colors[algo],
        alpha=0.2
    )


# =============================================================================
# 9. Figure labels
# =============================================================================

plt.xlabel(
    'Episode',
    fontsize=12,
    font={'family': 'Times New Roman'}
)

plt.ylabel(
    'Cumulative Success',
    fontsize=12,
    font={'family': 'Times New Roman'}
)


# =============================================================================
# 10. Legend
# =============================================================================

plt.legend(
    fontsize=10,
    loc='lower right'
)


# plt.grid(
#     linestyle='--',
#     alpha=0.5
# )


plt.tight_layout()


# =============================================================================
# 11. Save figure
# =============================================================================

# plt.savefig(
#     'C:/Users/37409/Desktop/Figures/Fig.8(b).pdf',
#     dpi=1200,
#     bbox_inches='tight'
# )

# plt.savefig(
#     'C:/Users/37409/Desktop/Figures/Fig.8(b).png',
#     dpi=1200,
#     bbox_inches='tight'
# )


# =============================================================================
# 12. Display figure
# =============================================================================

plt.show()
