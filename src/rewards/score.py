import numpy as np

from src.rewards.conj_chemistry import *



# Set the reward function based on the argument
reward_dict = {
    0 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    1 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    2 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    3 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    4 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    5 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    6 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    7 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    8 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    9 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    10 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    11 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    12 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    13 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    14 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    15 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    16 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    17 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    18 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    19 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
    20 : conj0_score,        # Conjecture XXX -- known counterexample with XXX nodes
}


def score_state_graph(args,state):
    # step 0: convert state to adj_matrix

    # step 1: convnert adj_matrix to G i.e networkx graph

    # step 2: compute score

    # step 3: optional reward shaping

    # step 4: return score, info, etc


    return