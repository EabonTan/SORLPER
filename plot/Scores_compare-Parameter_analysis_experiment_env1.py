
import os
import numpy as np
import warnings

from batch_Parameters_main2 import env_params

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 1. Experimental parameters
# ============================================================

num_episodes = env_params['num_trials']
num_runs = env_params['num_repeats']

# Environment 1 goal position
friend = np.array(
    env_params['agent_goal'],
    dtype=float
)

# Success threshold
success_threshold = 10.0

num_configs = 7


# ============================================================
# 2. Parameter configurations
# ============================================================

param_labels = [
    r'kappa1=0.01, kappa2=0.99',
    r'kappa1=0.1, kappa2=0.9',
    r'kappa1=0.2, kappa2=0.8',
    r'kappa1=0.5, kappa2=0.5',
    r'kappa1=0.8, kappa2=0.2',
    r'kappa1=0.9, kappa2=0.1',
    r'kappa1=0.99, kappa2=0.01'
]


# Environment 1 result directories
result_dirs = [
    'results_ParaAna - 0.0199',
    'results_ParaAna - 19',
    'results_ParaAna - 28',
    'results_ParaAna - 0.5',
    'results_ParaAna - 82',
    'results_ParaAna - 91',
    'results_ParaAna - 0.991'
]


# ============================================================
# 3. Base path
# ============================================================

base_path = (
    'E:/FILE/SORLPER_Nav/SORLPER'
)

output_dir = os.path.join(
    base_path,
    'Environment1_Parameter_Analysis_Results'
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================
# 4. Load navigation trajectory data
# ============================================================

def get_record(file_path, num_episodes, friend,
               success_threshold=10.0):

    """
    Load trajectory data from one independent run.

    Column 0: Navigation steps.
    Column 1: Success indicator.

    Success criterion:
    Euclidean distance between the final position
    and the goal position is smaller than the threshold.
    """

    load_dict = np.load(
        file_path,
        allow_pickle=True
    ).item()

    record = np.full(
        (num_episodes, 2),
        np.nan,
        dtype=float
    )

    for ep in range(1, num_episodes + 1):

        key = 'episode' + str(float(ep))

        if key not in load_dict:
            continue

        episode = np.asarray(
            load_dict[key],
            dtype=float
        )

        if episode.ndim != 2 or episode.shape[1] < 2:
            raise ValueError(
                f'Invalid trajectory format: {key}, '
                f'shape={episode.shape}'
            )

        if len(episode) == 0:
            raise ValueError(
                f'Empty trajectory: {key}'
            )

        xy = episode[:, :2]

        if not np.isfinite(xy).all():
            raise ValueError(
                f'Invalid coordinates in {key}'
            )

        # --------------------------------------------
        # 1. Navigation steps
        # --------------------------------------------

        steps = len(xy) - 1

        record[ep - 1, 0] = steps

        # --------------------------------------------
        # 2. Navigation success
        # --------------------------------------------

        last_location = xy[-1]

        distance = np.linalg.norm(
            last_location - friend[:2]
        )

        record[ep - 1, 1] = (
            1 if distance < success_threshold else 0
        )

    return record


# ============================================================
# 5. Load all configurations and independent runs
# ============================================================

print('\nLoading Environment 1 navigation trajectories...')

episode_success_results = {}
navigation_steps_results = {}

for j in range(num_configs):

    success_list = []
    steps_list = []

    for run in range(num_runs):

        file_path = os.path.join(
            base_path,
            result_dirs[j],
            f'agent_sequence_env{run}.npy'
        )

        try:

            record = get_record(
                file_path,
                num_episodes,
                friend,
                success_threshold
            )

            steps = record[:, 0]
            success = record[:, 1]

            # Check missing episodes
            if (
                np.isnan(steps).any()
                or np.isnan(success).any()
            ):

                missing_episodes = (
                    np.where(np.isnan(success))[0] + 1
                )

                raise ValueError(
                    f'Missing episode records: '
                    f'{missing_episodes.tolist()}'
                )

            success_list.append(success)
            steps_list.append(steps)

        except Exception as e:

            raise RuntimeError(
                f'Failed to load trajectory data:\n'
                f'{file_path}\n{e}'
            )

    episode_success_results[j] = np.vstack(
        success_list
    )

    navigation_steps_results[j] = np.vstack(
        steps_list
    )


# ============================================================
# 6. Calculate success rate
# ============================================================

success_rate_results = {}

for j in range(num_configs):

    success_data = episode_success_results[j]

    # Success rate of each independent run
    run_success_rates = (
        np.sum(success_data, axis=1)
        / num_episodes
        * 100.0
    )

    success_rate_results[j] = {
        'run_values': run_success_rates,
        'mean': np.mean(run_success_rates),
        'std': np.std(run_success_rates)
    }


# ============================================================
# 7. Calculate average navigation steps
# ============================================================

successful_steps_results = {}

for j in range(num_configs):

    steps_data = navigation_steps_results[j]

    success_data = episode_success_results[j]

    run_average_steps = []

    for run in range(num_runs):

        successful_mask = (
            success_data[run] == 1
        )

        successful_steps = (
            steps_data[run][successful_mask]
        )

        if len(successful_steps) == 0:

            raise ValueError(
                f'No successful episodes for '
                f'{result_dirs[j]}, run {run}'
            )

        # Average steps of successful episodes
        run_average_steps.append(
            np.mean(successful_steps)
        )

    run_average_steps = np.asarray(
        run_average_steps
    )

    successful_steps_results[j] = {
        'run_values': run_average_steps,
        'mean': np.mean(run_average_steps),
        'std': np.std(run_average_steps)
    }


# ============================================================
# 8. Print statistical results
# ============================================================

print('\n' + '=' * 100)
print('ENVIRONMENT 1: NAVIGATION PERFORMANCE')
print('=' * 100)

for j in range(num_configs):

    success_counts = np.sum(
        episode_success_results[j],
        axis=1
    )

    rate = success_rate_results[j]

    steps = successful_steps_results[j]

    print('\n' + '-' * 100)

    print(
        f'Parameter configuration: {param_labels[j]}'
    )

    print(
        f'Result directory: {result_dirs[j]}'
    )

    print(
        f'Successful navigations in each run: '
        f'{success_counts.astype(int).tolist()}'
    )

    print(
        f'Success rate in each run (%): '
        f'{np.round(rate["run_values"], 2).tolist()}'
    )

    print(
        f'Average success rate: '
        f'{rate["mean"]:.2f} ± {rate["std"]:.2f}%'
    )

    print(
        f'Average navigation steps: '
        f'{steps["mean"]:.2f} ± {steps["std"]:.2f}'
    )


# ============================================================
# 9. Save statistical results
# ============================================================

statistics_path = os.path.join(
    output_dir,
    'Environment1_Parameter_Statistics.txt'
)

with open(
    statistics_path,
    'w',
    encoding='utf-8'
) as f:

    f.write(
        'Environment 1 Parameter Analysis Results\n'
    )

    f.write('=' * 80 + '\n')

    f.write(
        f'Number of independent runs: {num_runs}\n'
    )

    f.write(
        f'Number of episodes per run: {num_episodes}\n'
    )

    f.write(
        f'Success criterion: Final distance to goal '
        f'< {success_threshold}\n'
    )

    f.write(
        'Average navigation steps are calculated '
        'using successful episodes only.\n\n'
    )

    for j in range(num_configs):

        rate = success_rate_results[j]
        steps = successful_steps_results[j]

        success_counts = np.sum(
            episode_success_results[j],
            axis=1
        )

        f.write('=' * 80 + '\n')

        f.write(
            f'Parameter configuration: {param_labels[j]}\n'
        )

        f.write(
            f'Result directory: {result_dirs[j]}\n\n'
        )

        f.write(
            'Successful navigations in each run:\n'
        )

        f.write(
            f'{success_counts.astype(int).tolist()}\n\n'
        )

        f.write(
            'Success rate (%):\n'
        )

        f.write(
            f'Each run: '
            f'{np.round(rate["run_values"], 2).tolist()}\n'
        )

        f.write(
            f'Mean ± Std: '
            f'{rate["mean"]:.2f} ± {rate["std"]:.2f}%\n\n'
        )

        f.write(
            'Average navigation steps '
            '(successful episodes):\n'
        )

        f.write(
            f'Each run: '
            f'{np.round(steps["run_values"], 2).tolist()}\n'
        )

        f.write(
            f'Mean ± Std: '
            f'{steps["mean"]:.2f} ± {steps["std"]:.2f}\n\n'
        )


# ============================================================
# 10. Save numerical data
# ============================================================

save_data = {}

for j in range(num_configs):

    save_data[
        f'episode_success_config_{j+1}'
    ] = episode_success_results[j]

    save_data[
        f'navigation_steps_config_{j+1}'
    ] = navigation_steps_results[j]

    save_data[
        f'success_rate_config_{j+1}'
    ] = success_rate_results[j]['run_values']

    save_data[
        f'average_steps_config_{j+1}'
    ] = successful_steps_results[j]['run_values']


np.savez(
    os.path.join(
        output_dir,
        'Environment1_Parameter_Data.npz'
    ),
    **save_data
)


# ============================================================
# 11. Final information
# ============================================================

print('\n' + '=' * 80)
print('ENVIRONMENT 1 STATISTICAL ANALYSIS COMPLETED.')
print('=' * 80)

print('\nOutput directory:')
print(output_dir)

print('\nGenerated files:')
print('1. Environment1_Parameter_Statistics.txt')
print('2. Environment1_Parameter_Data.npz')

