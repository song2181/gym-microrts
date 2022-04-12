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
import rl_utils as util
from smdp_qlearning_algo import SmdpAgent_Q
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import copy
from enums import ActionType,UnitType

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
    parser.add_argument('--epsilon', type=float, default=0.2,
        help='epsilon for exploration')
    parser.add_argument('--episode', type=int, default=30,
        help='episode')
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
    obs = []
    # rts_obs_space = rts_obs.shape(-1)
    rts_obs = rts_obs.squeeze()  ## 16*16*27
    for i in range(0,rts_obs.shape[0]):   ## x坐标
        for j in range(0,rts_obs.shape[1]):   ## y坐标
            hit_points = rts_obs[i][j][0:5]
            resource =  rts_obs[i][j][5:10]
            owner = rts_obs[i][j][10:13]
            unit_type = rts_obs[i][j][13:21]
            current_action = rts_obs[i][j][21:27]
            if unit_type[4] == 1 and owner[1]:  ## worker of player1
                obs.append(np.array([i,j]))
    return obs


def action_wrapper(pos, act_type, dir = 0, produce_type = 0, attack_pos = [0,0]):
    # act_type = [0,1,0,0,0,0]  ## move
    # move_dir = [0,0,0,0]      ## move parameter
    # act = np.zeros(shape=(1,256,78),dtype=np.int64)
    # move_dir[action] = 1
    # act[0][pos[0]*16+pos[1]] = act_type + move_dir + [0]*68
    # act = act.reshape(1,-1) 
    act = np.zeros(shape=(1,256,7),dtype=np.int64)
    act[0][pos[0]*16+pos[1]][0] = act_type
    if act_type != ActionType.Attack:
        act[0][pos[0]*16+pos[1]][act_type] = dir
        if act_type == ActionType.Produce:
            act[0][pos[0]*16+pos[1]][5] = produce_type
    else:
        act[0][pos[0]*16+pos[1]][6] = attack_pos[0]*16+attack_pos[1]
    act = act.reshape(1,-1)
    return act

def pos_forward(pos, action):
    if action == 0:
        pos[0] = max(0,pos[0]-1)
    elif action == 2:
        pos[0] = min(15,pos[0]+1)
    elif action == 1:
        pos[1] = min(15,pos[1]+1)
    elif action == 3:
        pos[1] = max(0,pos[1]-1)
    return pos



## new added for option methods
def step_option(options,cur_state): # cur_state is a position
    """Steps through an option until termnation, then returns the final
        observation, reward history, and finishing evaluation.
    """
    obs = [cur_state]
    acts = []
    rew  = []
    done = False
    # pos = copy.deepcopy(cur_state)
    step = 0
    while not done: # and not option.check_termination(obs[-1]):
        for agent,option in zip(obs[-1],options):
            action = option.act(agent)
            # action = 1
            if action is None or action == -1: # option was invalid
                continue
            else:
                if step%10 == 0:
                    envs.get_action_mask() 
                    act = action_wrapper(agent,ActionType.Move,action)
                    # pos = pos_forward(pos,action)
                    acts.append(act)
                    # time.sleep(2)
                else:
                    act = np.zeros(shape=(1,1792),dtype=np.int64)

                ob, re, do, info = envs.step(act)
                envs.render()
        if step%10 == 9:
            obs.append(obs_wrapper(ob))
            reward = sum([get_reward(i) for i in obs[-1]])
            rew.append(reward)
            done = sum([get_option_done(i,j.termination) for i,j in zip(obs[-1],options)])
        step+=1
    return obs, acts, rew, done, info


def get_option_done(obs,opt_terminal = terminal):
    '''
    if the pos is termination
    '''
    if type(obs)==np.ndarray:
        obs = obs.tolist()
    if type(opt_terminal) == np.ndarray:
        opt_terminal = opt_terminal.tolist()
    if obs == opt_terminal or obs in opt_terminal:
        return True
    else:
        return False

def step_action(action):
    step = 0
    while True:
        if step == 0:
            act = action
        else:
            act = np.zeros(shape=(1,1792),dtype=np.int64)
        ob, re, do, info = envs.step(act)
        envs.render()
        if step == 49:  ## produce need 50 step
            break
        step += 1
    return ob



def get_reward(obs):
    '''
    get to the terminal, the reward will be 1. else 0.
    '''
    reward = 0
    if type(obs)==np.ndarray:
        obs = obs.tolist()

    ## 是否添加中间奖励
    # reward = 10 - (abs(obs[0]-terminal[0]) + abs(obs[1]-terminal[1]))
    if obs == terminal:
        reward += 10
    return reward

def get_done(obs,opt_terminal = terminal):
    '''
    if the pos is termination
    '''
    if type(obs)==np.ndarray:
        obs = obs.tolist()
    if type(opt_terminal) == np.ndarray:
        opt_terminal = opt_terminal.tolist()
    if obs == opt_terminal or obs in opt_terminal:
        return True
    else:
        return False

# init two worker agent
def env_init(obs):
    action_mask = envs.get_action_mask()
    base_position = find_unit(obs,UnitType.Base)  # find the base: 1
    action = action_wrapper(base_position,ActionType.Produce,1,produce_type=3)  ## dir = east, type = worker
    obs_next = step_action(action)
    return obs_next

def find_unit(rts_obs,unittype): ## position rts_obs 1*16*16*27
    obs = np.array([],dtype=int)
    # rts_obs_space = rts_obs.shape(-1)
    rts_obs = rts_obs.squeeze()  ## 16*16*27
    for i in range(0,rts_obs.shape[0]):   ## x坐标
        for j in range(0,rts_obs.shape[1]):   ## y坐标
            hit_points = rts_obs[i][j][0:5]
            resource =  rts_obs[i][j][5:10]
            owner = rts_obs[i][j][10:13]
            unit_type = rts_obs[i][j][13:21]
            current_action = rts_obs[i][j][21:27]
            if unit_type[unittype] == 1 and owner[1]:  ## unittype of player1
                obs = np.append(obs,[i,j], axis = 0)
    return obs

if __name__ == "__main__":
    # env init
    envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.passiveAI for _ in range(1)],
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    envs.action_space.seed(0)
    print(envs.action_plane_space.nvec)
    nvec = envs.action_space.nvec
    envs.reset() ## np.array 1*16*16*27

    args = parse_args()
    experiment_name = f"{args.gym_id}__{args.exp_name}__{int(time.time())}"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    writer = SummaryWriter(f"runs/{experiment_name}")

    num_actions = 4   ## up right down left
    state_space = np.array([[i,j] for i in range(0,16) for j in range(0,16)]).astype(int)
    landmarks = [[1,4],[4,4],[4,7],[7,7]]
    options = util.create_options(landmarks)

    q_func = util.QTable(state_space,num_actions)
    agent_smdp = SmdpAgent_Q(q_func,options)
    total_step = 0
    total_option = 0


    for i in range(args.episode):
        envs.render()
        print(i)
        obs = envs.reset()

        ## init 2 agent
        obs = env_init(obs)

        cur_state = obs_wrapper(obs)## np.array 1*16*16*27

        # action_mask
        action_mask = envs.get_action_mask()  ##1*256*78
        action_mask = action_mask.reshape(-1, action_mask.shape[-1])## 256*78
        action_type_mask = action_mask[:,0:6]

        tot_td = 0
        done = False
        reward_record = []
        steps = 0
        options = 0
        max_options = 2000   ## max options in an episode

        for _ in range(max_options):

            #epsilon = np.max([0.1,1.-itr/(iterations/2.)]) # linear epsilon-decay
            opt  = [agent_smdp.pick_option_greedy_epsilon(i, eps=args.epsilon) for i in cur_state]
            states,actions,rewards,done,infos = step_option(opt,cur_state)
            for i in range(len(states[0])):
                tdes = util.q_learning_update_option_sequence(args.gamma, args.lr, \
                                            agent_smdp.q_func, [state[i] for state in states], \
                                            rewards, opt[i].identifier)
                tot_td   += np.sum(tdes)
                reward_record.append(rewards)
                cur_state = states[-1]
                steps += len(states)
                options += 1
            
            if get_done(cur_state[0]) or get_done(cur_state[1]):
                break
        total_step += steps
        total_option += options
        writer.add_scalar("options", total_option, i)
        writer.add_scalar("tot_td", tot_td, i)
        writer.add_scalar("steps", steps, i)
        # ret = util.discounted_return(reward_record,args.gamma) # CURRENTLY USING SWITCHING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # greedy_steps, greedy_choices, greedy_ret, greedy_success = util.switching_greedy_eval(agent_smdp,args.gamma,max_options,100)
    envs.close()
    writer.close()
