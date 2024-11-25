# parameter_grid.py

import itertools
import random

def generate_parameter_grid():
    """
    Generates a parameter grid for the CEM training script.
    Returns a list of unique parameter dictionaries.
    """


    # Default parameters that are fixed
    default_params = {

        'n': 10,
        'directed': False,
        'seed': 42,

        'check_every_k__best_graphs': 100,
        'print_every_k': 10,
        'print_on_improvement_of_reward': True,

        'logger': 'tensorboard',

        'reward_function': 'conjecture_2_1',

        'init_graph_type': 'empty',
        'iterations':50000,

        'stagnation_epochs': 130,


    }

    

    param_grid = {
        'learning_rate': [0.0001,0.0003,0.003,0.008, 0.001, 0.01],
        'batch_size': [32, 64, 128],
        'optimizer': ['SGD', 'Adam', ],
        'model': ['MLP'],   #, 'RNN', 'CNN'],
        'n_sessions': [ 250, 450,600],
        'percentile': [80,85,90,92,94,96,98],
        'super_percentile': [85, 88,90,93,96],
        'activation': ['relu', 'tanh', 'swish'],
        'init_method': ['xavier', 'kaiming','he'],
        'hidden_sizes': [[128, 128], [256, 256], [512, 512]],

        # Reward functions
        # 'reward_function': ['conjecture_2_1', 'conjecture_2_2', 'conjecture_2_3', 'conjecture_2_4', 'conjecture_2_5']
        # 'reward__is_tree': [True, False]
        # 'reward__is_connected_penalty': [True, False]
        # 'reward__is_tree_properties': [True, False]

        # Initial graph parameters
        # 'initial_graph_type': ['tree', 'full', 'empty'],
        # 'random_init_over_sessions': [True, False]
        #
    }

    # Model-specific parameters
    mlp_params = {
        'activation': ['relu', 'tanh', 'swish'],
        'normalize_input': [True, False],
        'batch_norm': [True, False],
        'dropout': [0.0,0.02,0.03, 0.07, 0.1],
        'hidden_size': [128, 256, 512],
    }

    rnn_params = {
        'hidden_size': [128, 256, 512],
        'num_layers': [1, 2, 3]
    }

    cnn_params = {
        'num_filters': [16, 32, 64],
        'kernel_size': [3, 5]
    }

    # Generate base parameter combinations
    base_keys, base_values = zip(*param_grid.items())
    base_combinations = [dict(zip(base_keys, v)) for v in itertools.product(*base_values)]

    # add default parameters to each combination
    for params in base_combinations:
        params.update(default_params)

    all_param_sets = []
    for base_params in base_combinations:
        model = base_params['model']
        if model == 'MLP':
            m_keys, m_values = zip(*mlp_params.items())
            m_combinations = [dict(zip(m_keys, v)) for v in itertools.product(*m_values)]
            for m_params in m_combinations:
                params = {**base_params, **m_params}
                all_param_sets.append(params)
        elif model == 'RNN':
            m_keys, m_values = zip(*rnn_params.items())
            m_combinations = [dict(zip(m_keys, v)) for v in itertools.product(*m_values)]
            for m_params in m_combinations:
                params = {**base_params, **m_params}
                all_param_sets.append(params)
        elif model == 'CNN':
            m_keys, m_values = zip(*cnn_params.items())
            m_combinations = [dict(zip(m_keys, v)) for v in itertools.product(*m_values)]
            for m_params in m_combinations:
                params = {**base_params, **m_params}
                all_param_sets.append(params)

    # Shuffle to ensure random sampling
    random.shuffle(all_param_sets)
    return all_param_sets
