from enum import Flag
import numpy as np
from numpy.random import choice
# if you want to record videos, install stable-baselines3 and use its `VecVideoRecorder`
# from stable_baselines3.common.vec_env import VecVideoRecorder

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
import argparse
import wandb
import torch
import time
from distutils.util import strtobool
import os
# import rl_utils as util
from rl_utils import *
from smdp_qlearning_algo import SmdpAgent_Q
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import random
import copy

terminal = [7,7]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
        help='the id of the gym environment')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')

    ## args for model
    parser.add_argument('--epsilon', type=float, default=0.1,
        help='epsilon for exploration')
    parser.add_argument('--iteration', type=int, default=1000,
        help='iteration')
    parser.add_argument('--gamma', type=float, default=0.9,
        help='discount for return')
    parser.add_argument('--lr', type=float, default=1./16.,
        help='the learning rate')
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    # args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    # args.batch_size = int(args.num_envs * args.num_steps)
    # args.minibatch_size = int(args.batch_size // args.n_minibatch)
    # args.num_updates = args.total_timesteps // args.batch_size
    # args.save_frequency = int(args.num_updates // 100)
    return args


# envs = VecVideoRecorder(envs, 'videos', record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sample(logits):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    return np.array([0 if sum(item)==0 else choice(range(len(item)), p=item/sum(item)) for item in logits]
    ).reshape(-1, 1)

# def sample(logits):
#     return np.array(
#         [choice(range(len(item)), p=softmax(item)) for item in logits]
#     ).reshape(-1, 1)

def obs_wrapper(rts_obs): ## position rts_obs 1*16*16*27
    agent_pos = {}
    mine_pos = []
    base_pos = []
    # rts_obs_space = rts_obs.shape(-1)
    rts_obs = rts_obs.squeeze()  ## 16*16*27
    for i in range(0,rts_obs.shape[0]):   ## x坐标
        for j in range(0,rts_obs.shape[1]):   ## y坐标
            hit_points = np.argmax(rts_obs[i][j][0:5])
            resource =  np.argmax(rts_obs[i][j][5:10])
            owner = rts_obs[i][j][10:13]
            unit_type = np.argmax(rts_obs[i][j][13:21])
            current_action = np.argmax(rts_obs[i][j][21:27])
            if unit_type == UNIT_TYPE['resource'] and resource>0:
                mine_pos.append([i,j])
            if unit_type == UNIT_TYPE['base'] and owner[1]:
                base_pos.append([i,j])
            if unit_type >= UNIT_TYPE['base'] and owner[1] and current_action == 0:  ## not base and resource of player1
                agent_pos[pos_to_idx([i,j])] = unit_type
    return agent_pos,mine_pos,base_pos


def action_wrapper(pos, action):
    # act_type = [0,1,0,0,0,0]  ## move
    # move_dir = [0,0,0,0]      ## move parameter
    # act = np.zeros(shape=(1,256,78),dtype=np.int64)
    # move_dir[action] = 1
    # act[0][pos[0]*16+pos[1]] = act_type + move_dir + [0]*68
    # act = act.reshape(1,-1)
    act_type = 1  ##move
    move_dir = action  
    act = np.zeros(shape=(1,256,7),dtype=np.int64)
    act[0][pos[0]*16+pos[1]]= [act_type , move_dir ,0,0,0,0,0]
    act = act.reshape(1,-1)
    return act


## new added for option methods
# def step_option(option,cur_state,step_count):
#     """Steps through an option until termnation, then returns the final
#         observation, reward history, and finishing evaluation.
#     """
#     obs = [cur_state]
#     acts = []
#     rew  = []
#     done = False
#     cur = 0
    
#     while not done: # and not option.check_termination(obs[-1]):
        
#         pos = obs[-1]
#         action = option.act(pos,cur)
#         cur += 1
#         # action = random.randint(0,3)

#         print(action)
#         if action is None: # option was invalid
#             break
#         else:
#             envs.render()
#             if step_count%10==0:
#                 # action_mask = envs.get_action_mask()
#                 # act = action_wrapper(pos,action)
#                 # act[0][17*7:18*7] = [1,1,0,0,0,0,0]
#                 # act[0][18*7:19*7] = [1,3,0,0,0,0,0]
#                 acts.append(action)
#                 print("act")
#             else:
#                 act= np.zeros(shape=8,dtype=np.int64)
#                 acts.append(act)
            
            
#             ob, re, do, info = envs.step([[acts[-1]]])
#             step_count+=1
            
            
#             # while (obs_wrapper(ob) == pos).all():
#             #     p = 0
#             obs.append(obs_wrapper(ob))
#             reward = get_reward(ob)
#             rew.append(reward)
#             done = get_done(ob)
#     return obs, acts, rew, done, step_count


def step_option(cur_option, action_mask, split_index):
    acts = []
    new_option = copy.deepcopy(cur_option)
    for idx, opt in cur_option.items():
        act = opt.policy[opt.cur_action]
        source_unit_action_mask = action_mask[idx]
        split_suam = np.split(source_unit_action_mask, split_index)
        mask = [split_suam[i-1][act[i]] for i in range(1,8)]
        # 1. 有动作 2 动作可执行 3 动作方向参数可执行
        if invalid_action(act, mask):
            new_option[idx].cur_action += 1
            acts.append(act)
            if act[1] == ACTION_TYPE['move']:
                new_pos = move(idx_to_pos(idx),dir=act[2])
                new_option[pos_to_idx(new_pos)] = new_option.pop(idx)
    ob, re, do, info = envs.step([acts])
    
    return ob, acts, re, do, new_option

def invalid_action(act, mask):
    if act[1] > 0 and act[1] < ACTION_TYPE['produce']: # move\harvest\return
        return mask[0] and mask[act[1]] 
    elif act[1] == ACTION_TYPE['produce']:# produce
        return mask[0] and mask[4] and mask[5]
    elif act[1] == ACTION_TYPE['attack']:# attack
        return mask[0] and mask[6]
    else:
        return True # NOOP
def get_reward(obs):
    '''
    get to the terminal, the reward will be 1. else 0.
    '''
    reward = 0
    if type(obs)==np.ndarray:
        obs = obs.tolist()
    if obs == terminal:
        reward += 1
    return reward

def get_done(obs):
    '''
    if the pos is termination
    '''
    if type(obs)==np.ndarray:
        obs = obs.tolist()
    if obs == terminal:
        return True
    else:
        return False

def check_options_termination(options):
    finished_option = {}
    new_option = copy.deepcopy(options)
    for idx,opt in options.items():
        if opt.check_termination(idx_to_pos(idx)):
            finished_option[idx] = new_option.pop(idx)
    return new_option,finished_option

if __name__ == "__main__":
    # env init
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        max_steps=2000,
        render_theme=1,
        ai2s=[microrts_ai.coacAI for _ in range(1)],
        map_path="maps/16x16/basesWorkers16x16.xml",
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    envs.action_space.seed(0)
    # print(envs.action_plane_space.nvec)
    envs.reset()## np.array 1*16*16*27
    nvec = envs.action_space.nvec
     

    args = parse_args()
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # writer = SummaryWriter(f"runs/{experiment_name}")
    # writer.add_text(
    #     "hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    # )
    num_actions = 2   ## up right down left
    state_space = np.array([[i,j] for i in range(0,16) for j in range(0,16)]).astype(int)
    landmarks = [[1,4],[4,4],[4,7],[7,7]]
    # options = create_options(landmarks)
    new_state = envs.reset()
    agent_pos,mine_pos,base_pos = obs_wrapper(new_state)## np.array 1*16*16*27
    options = create_options(idx_to_pos(list(agent_pos.keys())[0]),list(agent_pos.values())[0],mine_pos,base_pos)
    # produce_woker = Option()

    q_func = QTable(state_space,num_actions)
    agent_smdp = SmdpAgent_Q(q_func,options)
    split_index = [sum(envs.action_space.nvec.tolist()[1:a]) for a in range(2,8)]
    states = []
    reward_record = []
    actions = []
    cur_option = {}
    for i in range(args.iteration):
        envs.render()
        agent_pos,mine_pos,base_pos = obs_wrapper(new_state)## np.array 1*16*16*27
        print(i)
        # TODO: this numpy's `sample` function is very very slow.
        # PyTorch's `sample` function is much much faster,
        # but we want to remove PyTorch as a core dependency...
        action_mask = envs.get_action_mask()  ##1*256*78
        action_mask = action_mask.reshape(-1, action_mask.shape[-1])## 256*78

        tot_td = 0
        done = False

        steps = 0
        max_options = 200
        for cur_state_idx,unit_type in agent_pos.items():
            if cur_state_idx not in cur_option.keys():
        #epsilon = np.max([0.1,1.-itr/(itrations/2.)]) # linear epsilon-decay
                agent_smdp.options = create_options(idx_to_pos(cur_state_idx),unit_type, mine_pos,base_pos)
                opt  = agent_smdp.pick_option_greedy_epsilon(idx_to_pos(cur_state_idx), unit_type, eps=args.epsilon)
                cur_option[cur_state_idx] = opt

        new_state,action,reward,done,cur_option = step_option(cur_option, action_mask,split_index)


        reward_record.append(reward)
        states.append(new_state)
        actions.append(action)
        cur_option = check_options_termination(cur_option)
        if done:
            break
    envs.close()
    # writer.close()
