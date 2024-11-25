import argparse
import torch
import os
import json



from src.rl_algo.cem.train import train_CEM_graph_agent
from src.rl_algo.cem.utils import create_output_folder
from src.rewards.score import reward_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Train a reinforcement learning model using cross entropy method.")
    # >>>>>> Environment arguments <<<<<<
    parser.add_argument('--n', type=int, default=11, help='Number of vertices in the graph.')
    parser.add_argument('--directed', type=bool, default=False, help='Whether the graph is directed.')
    parser.add_argument('--reward_function', type=int, default=10, help='Reward function to use.')
    parser.add_argument('--seed', type=int, default=29092000, help='Seed for random number generators.')
    parser.add_argument('--init_graph', type=str, default='empty', help='Initial graph type.', choices=['empty', 'complete',"random_tree"])

    # >>>>>> Reward shaping arguments <<<<<<
    parser.add_argument('--reward_weight_is_tree_pos', type=float, default=0.5, help='Reward weight for tree reward.')
    parser.add_argument('--reward_weight_is_tree_neg', type=float, default=0.5, help='Reward weight for tree reward.')
    parser.add_argument('--reward__is_tree', type=bool, default=False, help='Whether to use tree reward.')
    parser.add_argument('--reward_weight_cyclomatic', type=float, default=0.01, help='Reward weight for connectedness reward.')
    parser.add_argument('--reward__use_cylomatic', type=bool, default=False, help='Whether to use connectedness reward.')
    parser.add_argument('--INF', type=float, default=10000, help='Large negative value for bad actions such as disconnected graphs.')

    # conj 9 --> 44 nodes
    # conj 10 --> 12 nodes

    # >>>>>> Cross Entropy Method arguments <<<<<<
    parser.add_argument('--iterations', type=int, default=1000000, help='Number of iterations for training.')
    parser.add_argument('--n_sessions', type=int, default=200, help='Number of new sessions per iteration.')
    parser.add_argument('--percentile', type=int, default=91, help='Percentile for selecting elite sessions.')
    parser.add_argument('--super_percentile', type=int, default=92, help='Percentile for selecting super sessions.')

    # >>>>>> Machine Learning Models <<<<<<
    parser.add_argument('--model', type=str, default='MLP', help='What ML model to use.')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[128, 64, 4], help='Hidden layer sizes for the MLP model.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the MLP model.')
    parser.add_argument('--batch_norm', type=bool, default=False, help='Whether to use batch normalization in the MLP model.')
    parser.add_argument('--activation', type=str, default='swish', help='Activation function for the MLP model.',choices = ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh'])
    parser.add_argument('--init_method', type=str, default='xavier', help='Initialization method for the MLP model.')

    # >>>>>> Training arguments <<<<<<
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use (SGD, Adam, RMSprop).')
    
    # >>>>>> Logging arguments <<<<<<
    parser.add_argument('--logger', type=str, default='tensorboard', choices=['wandb', 'tensorboard'], help='Logger to use.')
    parser.add_argument('--wandb_entity', type=str, default='dezaarna', help='Wandb entity.')
    parser.add_argument('--wandb_project', type=str, default='CEM_runs', help='Wandb project name.')
    parser.add_argument('--save_every_k_iters', type=int, default=100, help='Save best graphs every k iterations.')
    parser.add_argument('--print_every_k', type=int, default=5, help='Print progress every k iterations.')
    parser.add_argument('--print_on_improvement_of_reward', type=bool, default=True, help='Print progress when reward improves.')

    parser.add_argument('--current_idx', type=int, default=69, help='')
    parser.add_argument('--base_folder', type=str, default='output', help='Output folder for saving results.')

    # >>>>>> Cache arguments <<<<<<
    parser.add_argument('--reward_tolerance', type=float, default=0.01, help='Reward tolerance for caching solutions.')

    # >>>>>> Local Search arguments <<<<<<
    parser.add_argument('--use_local_search', type=bool, default=False, help='Whether to use local search.')
    parser.add_argument('--when_to_start_LS', type=int, default=10000, help='When to start local search.')
    parser.add_argument('--do_LS_every_k_iters', type=int, default=12, help='Do local search every k iterations.')
    parser.add_argument('--append_to_data', type=bool, default=True, help='Whether to append local search results to data.')

    # >>>>>> Stagnation detection arguments <<<<<<
    parser.add_argument('--stagnation_epochs', type=int, default=100, help='Number of epochs to detect stagnation.')

    # Normalize input
    parser.add_argument('--normalize_input', type=bool, default=False, help='Normalize input.')
    args = parser.parse_args()
    return args



if __name__ == "__main__":


    args = parse_args()

    # Print arguments
    print("\n\n\t\t Arguments:")
    for arg in vars(args):
        print(f"\t\t\t {arg}: {getattr(args, arg)}")

    # Set device (handle CUDA, MPS, or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")

    # Create output folder based on arguments
    output_folder = create_output_folder(args)

    args.output_folder = output_folder

    # Save all arguments to a JSON file
    args_path = os.path.join(output_folder, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Arguments saved to {args_path}")

    args.device = device
    print("\n\n\t\t Starting training on device: ", device)
    print("\n\n\t\t Output folder: ", output_folder)


    args.calc_score = reward_dict[args.reward_function]


    train_CEM_graph_agent(args)








