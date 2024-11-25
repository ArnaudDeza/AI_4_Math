import numpy as np
import time
import torch

from src.rewards.score import score_state_graph



def play_game(args, actions,state_next,states,prob, step, total_score):
    infos = {}
    for i in range(args.n_sessions):
        if np.random.rand() < prob[i]:
            action = 1
        else:
            action = 0
        actions[i][step-1] = action
        state_next[i] = states[i,:,step-1]

        if (action > 0):
            state_next[i][step-1] = action
        state_next[i][args.MYN + step-1] = 0
        if (step < args.MYN):
            state_next[i][args.MYN + step] = 1            
        #calculate final score
        terminal = step == args.MYN
        if terminal:
            total_score[i],info = score_state_graph(args,state_next[i])
            infos["session_{}".format(i)] = info
    
        # record sessions 
        if not terminal:
            states[i,:,step] = state_next[i]
        
    return actions, state_next,states, total_score, terminal,infos



def generate_session(args,agent):    
    """
    Play n_session games using agent neural network.
    Terminate when games finish 
    
    Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
    """
    states =  np.zeros([args.n_sessions, args.observation_space, args.len_game], dtype=int)
    actions = np.zeros([args.n_sessions, args.len_game], dtype = int)
    state_next = np.zeros([args.n_sessions,args.observation_space], dtype = int)
    prob = np.zeros(args.n_sessions)
    states[:,args.MYN,0] = 1
    total_score = np.zeros([args.n_sessions])
    step,pred_time,play_time = 0, 0, 0
    
    while (True):
        step += 1        

        tic = time.time()
        state_tensor = torch.from_numpy(states[:, :, step - 1]).float().to(args.device)
        prob = agent(state_tensor).detach().cpu().numpy()  # Detach and move predictions to CPU
        pred_time += time.time()-tic
        tic = time.time()

        actions, state_next, states, total_score, terminal,infos = play_game( args, actions,state_next, states,prob, step, total_score)
        play_time += time.time()-tic
        
        if terminal:
            break
    return (states, actions, total_score), infos