
env_params = {
              'num_trials': 500,
              'max_steps': 300,
              'num_repeats': 10,
              'state_dim': 6,
              'action_dim': 36,
              'agent_start': [50, 30],
              'agent_goal': [450, 450],
              'save_path': 'results',  # results1 or results2
              'env': 1     # 1 or 2
              }

agent_params = {
                'bSOM': True,
                'SOM_alpha': 0.01,
                'SOM_sigma': .1,
                'SOM_sigma_const': .1,
                'SOM_delta': False,
                'Q_alpha': .9,
                'w_decay': 10,
                'TD_decay': 1,
                'SOM_size': 15,
                'learning_rate': 0.001,
                'discount_rate': 0.98,
                'hidden_dim': 128,
                'start_epsilon': 1,
                'final_epsilon': 0.001,
                'target_update': 1,
                'buffer_size': 10000,
                'minimal_size': 64,
                'batch_size': 64,
                'w1': 0.01,
                'w2': 0.99,
                'zeta': 2.2,
                'w1_w2_lr': 0.001,
                'dynamic_w1_w2': False,
                'actor_lr': 0.0001,
                'critic_lr': 0.001,
                'tau': 0.001,
                'noise_theta': 0.15,
                'noise_sigma': 0.2,
                }
