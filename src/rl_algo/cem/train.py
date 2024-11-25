from torch import nn
import numpy as np
from time import time
import wandb
import torch

from src.rl_algo.cem.game import generate_session


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

    # Step 3: Initialize logger
    if args.logger == 'wandb':
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        wandb.watch(model)
    elif args.logger == 'tensorboard':
        writer = SummaryWriter(log_dir=output_folder)

    # Step 4: Initialize variables
    # Generate initial buffers for super states, actions, and rewards
    super_states =  np.empty((0,len_game,observation_space), dtype = int)
    super_actions = np.array([], dtype = int)
    super_rewards = np.array([])
    # Timing variables
    sessgen_time, fit_time, score_time = 0, 0, 0
    # Initialize variables for stagnation detection and best solutions cache
    best_reward_list, best_solutions_cache, stagnation_counter = [], [], 0

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
            elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile) #pick the sessions to learn from
            select1_time = time.time()-tic

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
            train_network(model, optimizer, train_loader,device=device)
            fit_time = time.time() - tic


            # 6. Update super sessions
            tic = time.time()
            super_states = np.array([s[0] for s in super_sessions])
            super_actions = np.array([s[1] for s in super_sessions])
            super_rewards = np.array([s[2] for s in super_sessions])
            mean_all_reward = np.mean(rewards_batch[-100:])  # Mean reward for the best 100 sessions
            mean_best_reward = np.mean(super_rewards)  # Mean reward of the surviving sessions
            score_time = time.time() - tic
    
    
    
    
    
    
    
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








sessgen_time = 0
fit_time = 0
score_time = 0


myRand = random.randint(0,1000) #used in the filename

for i in range(1000000): #1000000 generations should be plenty
    #generate new sessions
    #performance can be improved with joblib
    tic = time.time()
    sessions = generate_session(args,model) #change 0 to 1 to print out how much time each step in generate_session takes 
    sessgen_time = time.time()-tic
    tic = time.time()
    
    states_batch = np.array(sessions[0], dtype = int)
    actions_batch = np.array(sessions[1], dtype = int)
    rewards_batch = np.array(sessions[2])
    states_batch = np.transpose(states_batch,axes=[0,2,1])
    
    states_batch = np.append(states_batch,super_states,axis=0)

    if i>0:
        actions_batch = np.append(actions_batch,np.array(super_actions),axis=0)    
    rewards_batch = np.append(rewards_batch,super_rewards)
        
    randomcomp_time = time.time()-tic 
    tic = time.time()

    elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile) #pick the sessions to learn from
    select1_time = time.time()-tic

    tic = time.time()
    super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile) #pick the sessions to survive
    select2_time = time.time()-tic
    
    tic = time.time()
    super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
    super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
    select3_time = time.time()-tic
    
    tic = time.time()
    
    train_data = torch.from_numpy(np.column_stack((elite_states, elite_actions)))
    train_data = train_data.to(torch.float)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32)
    train_network(model, optimizer, train_loader)
    fit_time = time.time()-tic
    
    tic = time.time()
    
    super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
    super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
    super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
    
    rewards_batch.sort()
    mean_all_reward = np.mean(rewards_batch[-100:])    
    mean_best_reward = np.mean(super_rewards)    

    score_time = time.time()-tic
    
    print("\n" + str(i) +  ". Best individuals: " + str(np.flip(np.sort(super_rewards))))
    
    #uncomment below line to print out how much time each step in this loop takes. 
    print(    "Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time)) 
    
    
    display_graph(state_to_graph(super_actions[0])[0])

    if (i%20 == 1): #Write all important info to files every 20 iterations
        with open('best_species_pickle_'+str(myRand)+'.txt', 'wb') as fp:
            pickle.dump(super_actions, fp)
        with open('best_species_txt_'+str(myRand)+'.txt', 'w') as f:
            for item in super_actions:
                f.write(str(item))
                f.write("\n")
        with open('best_species_rewards_'+str(myRand)+'.txt', 'w') as f:
            for item in super_rewards:
                f.write(str(item))
                f.write("\n")
        with open('best_100_rewards_'+str(myRand)+'.txt', 'a') as f:
            f.write(str(mean_all_reward)+"\n")
        with open('best_elite_rewards_'+str(myRand)+'.txt', 'a') as f:
            f.write(str(mean_best_reward)+"\n")
    if (i%200==2): # To create a timeline, like in Figure 3
        with open('best_species_timeline_txt_'+str(myRand)+'.txt', 'a') as f:
            f.write(str(super_actions[0]))
            f.write("\n")