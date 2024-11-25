import numpy as np
import networkx as nx


# Imports
from src.rewards.utils import state_to_graph



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
    # step 0: convert state to adj_matrix and G i.e networkx graph
    G, adjMatG, edgeListG, Gdeg = state_to_graph(args, state, directed = args.directed)
    
    # step 1: compute score
    score, information = reward_dict[args.reward_function](G, adjMatG, args.INF)

    # step 2: optional reward shaping:  If the graph is a tree, add a reward
    if args.reward__is_tree:
        if nx.is_tree(G):
            score += args.reward_weight_is_tree_pos
            information['is_tree'] = True
            information['cyclomatic_number'] = 0 

        else:
            score -= args.reward_weight_is_tree_neg
            information['is_tree'] = False
            # Step 2a: If we know that we do want a tree, look at tree properties that we may want:
            if args.reward__use_cylomatic:
                # Compute the cyclomatic number (number of cycles)
                num_components = nx.number_connected_components(G)
                cyclomatic_number = information['num_edges'] - information['num_nodes'] + num_components
                information['cyclomatic_number'] = cyclomatic_number
                # you want to minimize this as tree's have a cyclomatic number of 0
                score -= args.reward_weight_cyclomatic * cyclomatic_number
            else:
                information['cyclomatic_number'] = -1*args.INF

    # Update the information dictionary with final score used
    information['final_reward'] = score
    return score, information