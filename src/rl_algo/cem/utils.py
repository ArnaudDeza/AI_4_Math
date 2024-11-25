import numpy as np
import torch 
import random

import networkx as nx 
import matplotlib.pyplot as plt 
import datetime
import json
import os

# Imports
from src.models.cem.mlp import MLP
from src.models.cem.gnn import GNN
from src.models.cem.cnn import CNN
from src.models.cem.lr import LR

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def state_to_graph1(args, state, directed=False):
    """
    Efficiently constructs the graph from a given state and returns the NetworkX graph,
    adjacency matrix, edge list, and degree sequence.

    Args:
        args: An object with attributes like `n` (number of nodes).
        state: A 1D array representing the presence (1) or absence (0) of edges in the graph.
        directed: Whether the graph should be directed.

    Returns:
        G (networkx.Graph): The constructed NetworkX graph.
        adjMatG (numpy.ndarray): The adjacency matrix of the graph.
        edgeListG (list of lists): The neighbor list representation of the graph.
        Gdeg (numpy.ndarray): The degree sequence of the graph.
    """
    n = args.n
    state = np.asarray(state, dtype=np.int8)

    # Generate the upper triangle indices of the adjacency matrix
    triu_indices = np.triu_indices(n, k=1)
    
    # Fill the adjacency matrix
    adjMatG = np.zeros((n, n), dtype=np.int8)
    adjMatG[triu_indices] = state
    adjMatG += adjMatG.T  # Symmetrize for undirected graph

    # Generate degree sequence directly from the adjacency matrix
    Gdeg = adjMatG.sum(axis=1)

    # Create neighbor list
    edgeListG = [np.flatnonzero(adjMatG[i]).tolist() for i in range(n)]

    # Create a NetworkX graph
    if directed:
        G = nx.from_numpy_array(adjMatG, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(adjMatG, create_using=nx.Graph)

    return G, adjMatG, edgeListG, Gdeg



import networkx as nx
import numpy as np

def state_to_graph(args, state, directed=False):
    """
    Constructs the graph G from the given state and returns the NetworkX graph,
    adjacency matrix, edge list, and degree sequence.
    
    Args:
        args: An object with attributes like `n` (number of nodes).
        state: A list or array representing the edge states (1 for edge, 0 for no edge).
        directed: Whether the graph should be directed.

    Returns:
        G (networkx.Graph): The constructed NetworkX graph.
        adjMatG (numpy.ndarray): The adjacency matrix of the graph.
        edgeListG (numpy.ndarray): The neighbor list representation of the graph.
        Gdeg (numpy.ndarray): The degree sequence of the graph.
    """
    # Initialize adjacency matrix, edge list, and degree sequence
    adjMatG = np.zeros((args.n, args.n), dtype=np.int8)  # Adjacency matrix
    edgeListG = np.zeros((args.n, args.n), dtype=np.int8)  # Neighbor list
    Gdeg = np.zeros(args.n, dtype=np.int8)  # Degree sequence

    # Populate adjacency matrix and edge list
    count = 0
    for i in range(args.n):
        for j in range(i + 1, args.n):
            if state[count] == 1:
                adjMatG[i][j] = 1
                adjMatG[j][i] = 1
                edgeListG[i][Gdeg[i]] = j
                edgeListG[j][Gdeg[j]] = i
                Gdeg[i] += 1
                Gdeg[j] += 1
            count += 1

    # Create a NetworkX graph from the adjacency matrix
    if directed:
        G = nx.DiGraph(adjMatG)
    else:
        G = nx.Graph(adjMatG)

    return G, adjMatG, edgeListG, Gdeg


def initialize_model(args, MYN, device, seed):
    # Step 0 : seed everything
    seed_everything(seed)

    # Step 1: Choose Model
    if args.model == "MLP":
        model_args = {
            'input_size': 2 * MYN,
            'hidden_sizes': args.hidden_sizes,
            'dropout': args.dropout,
            'batch_norm': args.batch_norm,
            'activation': args.activation,
            'init_method': args.init_method
            }
        model = MLP(model_args).to(device)
    elif args.model == "LR":
        model_args = {

        }
        model = LR(model_args).to(device)
    elif args.model == "GNN":
        model_args = {

        }
        model = GNN(model_args).to(device)
    elif args.model == "CNN":
        model_args = {

        }
        model = CNN(model_args).to(device)

    # Step 2: Select optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    return model, optimizer



def display_graph(adjMatG):
    print("Best adjacency matrix in current step:")
    print(adjMatG)

    G = nx.convert_matrix.from_numpy_array(adjMatG)

    plt.clf()
    nx.draw_circular(G)

    plt.axis('equal')
    plt.draw()
    plt.pause(0.001)
    plt.show()



def create_output_folder(args):
    directed= "directed" if args.directed else "undirected"
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{args.base_folder}/{args.reward_function}__{directed}_n_{args.n}_{args.model}/cem_run__seed_{args.seed}_{timestamp}_id_{args.current_idx}"
    output_folder = os.path.join("results", folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder