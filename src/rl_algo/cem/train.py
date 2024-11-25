from torch import nn
import numpy as np
import time
import wandb
import torch
import random
import networkx as nx
from torch.utils.tensorboard import SummaryWriter


from src.rl_algo.cem.game import generate_session
from src.rl_algo.cem.utils import seed_everything, initialize_model
from src.rl_algo.cem.select import select_elites, select_super_sessions

def train_network(args, model, optimizer, train_loader, num_epochs=1):
    ''' Updates the model parameters (in place) using the given optimizer object.  Returns `None`. '''
    criterion,  pbar = nn.BCELoss(),  range(num_epochs)
    for i in pbar:
        for k, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(args.device)
            batch_x = batch_data[:, :-1]
            batch_y = batch_data[:, -1]
            model.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y.unsqueeze(1))
            loss.backward() 
            optimizer.step()


def action_to_adj_matrix(args,action):
    adjMatG = np.zeros((args.n,args.n),dtype=np.int8) #adjacency matrix determined by the state
    count = 0
    for i in range(args.n):
        for j in range(i+1,args.n):
            if action[count] == 1:
                adjMatG[i][j] = 1
                adjMatG[j][i] = 1
            count += 1
    return adjMatG




def train_CEM_graph_agent(args):

    # Step 0: Set random seed for reproducibility
    seed_everything(args.seed)

    # Step 1: Initialize graph parameters
    N = args.n
    MYN = int(N * (N - 1) / 2) if args.directed == False else N * (N - 1)
    observation_space,len_game = 2 * MYN, MYN
    state_dim = (observation_space,)
    args.state_dim, args.observation_space = state_dim, observation_space
    args.len_game, args.MYN = len_game, MYN

    # Step 2: Initialize model and optimizer
    model, optimizer = initialize_model(args, MYN, args.device,args.seed)
    model = model.to(args.device)

    # Step 3: Initialize logger
    if args.logger == 'wandb':
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        wandb.watch(model)
    elif args.logger == 'tensorboard':
        writer = SummaryWriter(log_dir=args.output_folder)

    # Step 4: Initialize variables
    # Generate initial buffers for super states, actions, and rewards
    super_states =  np.empty((0,len_game,observation_space), dtype = int)
    super_actions = np.array([], dtype = int)
    super_rewards = np.array([])
    # Timing variables
    sessgen_time, fit_time, score_time = 0, 0, 0
    # Initialize variables for stagnation detection and best solutions cache
    best_reward_list_every_iter,best_reward_list_overall, best_solutions_cache, stagnation_counter = [],[], [], 0
    mean_all_rewards_list, mean_super_rewards_list, mean_elite_rewards_list = [], [], []
    best_adj_matrix_list = []
    best_reward = -np.inf
    iter_since_valid_counter_example = 0

    sessgen_times,randomcomp_times,select1_times,select2_times,select3_times,fit_times,score_times = [],[],[],[],[],[],[]


    FINISHED = False
    try:
        # Optimized training loop
        for i in range(args.iterations):


            # 1. Generate new sessions (Parallelizing this can be beneficial)
            tic = time.time()
            sessions,session_stats = generate_session(args, model)  # Set verbose=1 for debugging timing
            sessgen_time = time.time() - tic

            # 2. Extract state, action, and reward batches
            tic = time.time()
            states_batch = np.array(sessions[0], dtype=int)
            actions_batch = np.array(sessions[1], dtype=int)
            rewards_batch = np.array(sessions[2])
            states_batch = np.transpose(states_batch, axes=[0, 2, 1])
            states_batch = np.append(states_batch,super_states,axis=0)
            if i>0:
                actions_batch = np.append(actions_batch,np.array(super_actions),axis=0)    
            rewards_batch = np.append(rewards_batch,super_rewards)
            randomcomp_time = time.time()-tic 
            
            # 3. Select elite sessions based on percentile
            tic = time.time()
            elite_states, elite_actions, elite_rewards = select_elites(args,states_batch, actions_batch, rewards_batch, percentile=args.percentile) #pick the sessions to learn from
            select1_time = time.time()-tic

            mean_elite_reward = np.mean(elite_rewards)  # Mean reward of the elite sessions


            # 4. Select super sessions to survive, using a diverse selection strategy
            tic = time.time()
            super_sessions = select_super_sessions(args, states_batch, actions_batch, rewards_batch, percentile=90)
            select2_time = time.time() - tic


            # Sort super sessions by rewards in descending order
            tic = time.time()
            super_sessions = sorted(zip(super_sessions[0], super_sessions[1], super_sessions[2]), key=lambda x: x[2], reverse=True)
            select3_time = time.time() - tic


            # 5. Train the model on elite sessions
            tic = time.time()
            train_data = torch.from_numpy(np.column_stack((elite_states, elite_actions))).float()
            train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
            train_network(args,model, optimizer, train_loader)
            fit_time = time.time() - tic



            # 6. Update super sessions
            tic = time.time()
            super_states = np.array([s[0] for s in super_sessions])
            super_actions = np.array([s[1] for s in super_sessions])
            super_rewards = np.array([s[2] for s in super_sessions])
            mean_all_reward = np.mean(rewards_batch[-100:])  # Mean reward for the best 100 sessions
            mean_best_reward = np.mean(super_rewards)  # Mean reward of the surviving sessions
            score_time = time.time() - tic

            # Update the best known candidate counter-example
            best_reward_this_iter = np.max(super_rewards)
            if best_reward_this_iter > best_reward:
                print("\t\t >> Found new best reward: {} improved from {} -- {} improvement".format(best_reward_this_iter, best_reward, best_reward_this_iter-best_reward))
                best_reward = best_reward_this_iter


            # 7. Logging -- Append values to list for plotting later
            best_reward_list_every_iter.append(best_reward_this_iter)
            best_reward_list_overall.append(best_reward)
            mean_all_rewards_list.append(mean_all_reward)               # Mean reward for the best 100 sessions
            mean_super_rewards_list.append(mean_best_reward)            # Mean reward of the surviving sessions
            mean_elite_rewards_list.append(mean_elite_reward)           # Mean reward of the elite sessions
            sessgen_times.append(sessgen_time)                          # Time taken to generate sessions
            randomcomp_times.append(randomcomp_time)                    # Time taken to extract states, actions, and rewards
            select1_times.append(select1_time)                          # Time taken to select elite sessions
            select2_times.append(select2_time)                          # Time taken to select super sessions
            select3_times.append(select3_time)                          # Time taken to sort super sessions
            fit_times.append(fit_time)                                  # Time taken to train the model
            score_times.append(score_time)                              # Time taken to update super sessions
            best_adj_matrix_list.append(super_actions[0]) # use action_to_adj_matrix(args,super_actions[0]) to get adj matrix



            # 9. Periodically save results
            if i % args.save_every_k_iters == 0:

                # Save list of scalars and adj matrices
                np.savez('data_lists.npz',
                     best_reward_list_every_iter=np.array(best_reward_list_every_iter),
                    best_reward_list_overall=np.array(best_reward_list_overall),
                    mean_all_rewards_list=np.array(mean_all_rewards_list),
                    mean_super_rewards_list=np.array(mean_super_rewards_list),
                    mean_elite_rewards_list=np.array(mean_elite_rewards_list),
                    sessgen_times=np.array(sessgen_times), randomcomp_times=np.array(randomcomp_times),
                    select1_times=np.array(select1_times), select2_times=np.array(select2_times), select3_times=np.array(select3_times),
                    fit_times=np.array(fit_times), score_times=np.array(score_times),
                    best_adj_matrix_list=np.array(best_adj_matrix_list)
                                                  )
                


            # 10. Check for termination conditions --> best reward > 0 or i == args.iterations - 1
            if i == args.iterations - 1:                # Hit max iterations
                FINISHED = True
                pass
            if best_reward > 0 and iter_since_valid_counter_example > 25:   # Found best counter-example but going for 25 more iterations
                iter_since_valid_counter_example += 1
                pass
            if best_reward > 0 and iter_since_valid_counter_example == 0:  # Found best counter-example for the first time
                print("\n\n\n\t\t >>> Found a valid counter-example with reward: {} at iteration: {}".format(best_reward, i))
                iter_since_valid_counter_example += 1

                adjmat_counter_example = action_to_adj_matrix(args,super_actions[0])

                # Save the best counter-example
                np.savez( 'valid_counter_example_score_{:.5f}.npz'.format(best_reward),
                    adj_mat =  adjmat_counter_example,      action = super_actions[0],  reward = np.array([best_reward]) )
                

                pass

            # 11. Priniting
            print("\t >>> Iter {}:  Best reward {} \t Mean all / super / elite rewards: {} / {} / {} ".format(i, best_reward_this_iter, mean_all_reward, mean_best_reward, mean_elite_reward))
            #print("\t >>> Iter {}: . Best individuals: {}".format(i, str(np.flip(np.sort(super_rewards)))))
            #print(    "Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time)) 
    

            if FINISHED:

                # Save list of scalars and adj matrices
                np.savez('data_lists.npz',
                     best_reward_list_every_iter=np.array(best_reward_list_every_iter),
                    best_reward_list_overall=np.array(best_reward_list_overall),
                    mean_all_rewards_list=np.array(mean_all_rewards_list),
                    mean_super_rewards_list=np.array(mean_super_rewards_list),
                    mean_elite_rewards_list=np.array(mean_elite_rewards_list),
                    sessgen_times=np.array(sessgen_times), randomcomp_times=np.array(randomcomp_times),
                    select1_times=np.array(select1_times), select2_times=np.array(select2_times), select3_times=np.array(select3_times),
                    fit_times=np.array(fit_times), score_times=np.array(score_times),
                    best_adj_matrix_list=np.array(best_adj_matrix_list)
                                                  )
                    
                     
    
    except Exception as e:
        print(f"\n\n\n \t\t > > > An error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Ensure resources are cleaned up
        if args.logger == 'wandb':
            wandb.finish()
        elif args.logger == 'tensorboard':
            writer.close()




