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



