# Shared utilities for testing various RL schemes on the Sutton Room World
import datetime
import pickle as pkl
import os.path
import numpy as np
import matplotlib.pyplot as plt
from smdp_qlearning_algo import *
import copy



ACTION_TYPE = {
    "NOOP":0,
    "move":1,
    "harvest":2,
    "return":3,
    "produce":4,
    "attack":5
}

DIR_TO_IDX = {
    "north":0,
    "east":1,
    "south":2,
    "west":3
}

PRODUCE_TYPE = {
    "resource":0,
    "base":1,
    "barrack":2,
    "worker":3,
    "light":4,
    "heavy":5,
    "ranged":6
}

UNIT_TYPE = {
    "none":0,
    "resource":1,
    "base":2,
    "barrack":3,
    "worker":4,
    "light":5,
    "heavy":6,
    "ranged":7
}

render_frame ={
    "move":10,
    "harvest":20,
    "return":10,
    "produce":50,
    "attack":5
}

resource_need = {
    "base":0,
    "barrack":5,
    "worker":1,
    "light":4,
    "heavy":4,
    "ranged":4,
}

IDX_TO_PRODUCE = dict(zip(PRODUCE_TYPE.values(), PRODUCE_TYPE.keys()))
IDX_TO_DIR = dict(zip(DIR_TO_IDX.values(), DIR_TO_IDX.keys()))
IDX_TO_ACTION = dict(zip(ACTION_TYPE.values(), ACTION_TYPE.keys()))
IDX_TO_UNIT = dict(zip(UNIT_TYPE.values(), UNIT_TYPE.keys()))

class QTable():
    """Class for storing q-values in a table.
    """
    def __init__(self,state_space,num_actions):
        self.num_actions = num_actions
        self.state_space = state_space
        self.table = {}
        for s in state_space:
            # Q-value is called by q_func.table[str(s)][a], where:
            #   q_func is a QTable object
            #   s is the agent position as an ndarray
            #   a is the index of the action
            self.table[str(s)] = np.zeros(num_actions)
            
    def __call__(self,state):
        """Returns the set of q-values stored for the given state.
        """
        try:
            qs = self.table[str(state)]
        except KeyError:
            qs = np.zeros(self.num_actions)
            #print("WARNING: KeyError in Q-function. Returning zeros.")
        return qs
    
    def update_table(self,state,q,action=None):
        if action is None: # if no action is specified, q should be an array of length num_actions
            assert(len(q)==self.num_actions)
            self.table[str(state)] = q
        else:
            self.table[str(state)][action] = q
    
class QTable_Numpy():
    """Class for storing q-values in a table.
    """
    def __init__(self,dimensions,num_actions):
        self.num_actions = num_actions
        self.dimensions = dimensions
        self.table = np.zeros(dimensions+(num_actions,))
            
    def __call__(self,state):
        """Returns the set of q-values stored for the given state.
        """
        try:
            qs = self.table[tuple(state)]
        except IndexError:
            qs = np.zeros(self.num_actions)
            print("WARNING: IndexError in Q-function. Returning zeros.")
        return qs
    
    def update_table(self,state,q,action=None):
        if action is None: # if no action is specified, q should be an array of length num_actions
            assert(len(q)==self.num_actions)
            self.table[tuple(state)] = q
        else:
            self.table[tuple(state)][action] = q

def create_options(agent_pos,unit_type, mine_pos,base_pos,my_state,op_state,action_mask,resource,mapp):
    options = []
    options.extend(create_transport_option(agent_pos,unit_type,base_pos,mine_pos,mapp))#1
    options.extend(create_produce_worker_option(agent_pos,my_state,len(base_pos)))#2
    options.extend(create_heavy_option(agent_pos))#3
    options.extend(create_light_option(agent_pos))#4
    options.extend(create_ranged_option(agent_pos))#5
    options.extend(create_attack_base_option(agent_pos,unit_type,op_state,mapp))#6
    options.extend(create_barrack_option(agent_pos,unit_type,resource,mapp))#7
    options.extend(create_return_option(agent_pos,unit_type,base_pos, mapp))#8
    # options.extend(create_attack_closest_option(agent_pos,unit_type,op_state,mapp))
    options.extend(create_NOOP_options(agent_pos))#99
    # options.extend(create_random_move_options(agent_pos,action_mask))#98
    
    return options

def create_random_move_options(agent_pos,action_mask):
    valid_state = [i for i in range(0,256)]  # all states
    terminate = []
    policy = []
    valid_dir = [i for i in range(0,4) if action_mask[1][i] == 1]
    if valid_dir == []:
        valid_state = []
    else:
        dir = np.random.choice(valid_dir)
        policy.append(np.array([pos_to_idx(agent_pos),ACTION_TYPE['move'],dir,0,0,0,0,0]))
    option = Option(policy,valid_state,terminate,len(policy),identifier=98)
    return [option]

def create_NOOP_options(agent_pos):
    valid_state = [i for i in range(0,256)]  # all states
    terminate = []
    policy = [np.array([pos_to_idx(agent_pos),ACTION_TYPE['NOOP'],0,0,0,0,0,0])]
    option = Option(policy,valid_state,terminate,len(policy),identifier=99)
    return [option]


def create_barrack_option(agent_pos,unit_type,resource,mapp):
    from A_star import get_move_policy
    valid_state = [i for i in range(0,256)]  # all states
    terminate = []
    if resource < 4 or mapp[1][3] == 1:
        valid_state = []
    policy = get_move_policy(agent_pos,[1,3],mapp)
    policy.append(np.array([pos_to_idx([1,3]),ACTION_TYPE['produce'],0,0,0,DIR_TO_IDX['north'],PRODUCE_TYPE['barrack'],0]))
    option = Option(policy,valid_state,terminate,len(policy),identifier=7)
    return [option]

def create_heavy_option(agent_pos):
    
    valid_state = [i for i in range(0,256)]  # all states
    terminate = []
    policy = [np.array([pos_to_idx(agent_pos),ACTION_TYPE['produce'],0,0,0,DIR_TO_IDX['south'],PRODUCE_TYPE['heavy'],0])]
    option = Option(policy,valid_state,terminate,len(policy),identifier=3)
    return [option]

def create_light_option(agent_pos):
    
    valid_state = [i for i in range(0,256)]  # all states
    terminate = []
    policy = [np.array([pos_to_idx(agent_pos),ACTION_TYPE['produce'],0,0,0,DIR_TO_IDX['south'],PRODUCE_TYPE['light'],0])]
    option = Option(policy,valid_state,terminate,len(policy),identifier=4)
    return [option]

def create_ranged_option(agent_pos):
    
    valid_state = [i for i in range(0,256)]  # all states
    terminate = []
    policy = [np.array([pos_to_idx(agent_pos),ACTION_TYPE['produce'],0,0,0,DIR_TO_IDX['south'],PRODUCE_TYPE['ranged'],0])]
    option = Option(policy,valid_state,terminate,len(policy),identifier=5)
    return [option]

def create_attack_base_option(pos,unit_type,op_state,mapp):
    valid_state = [i for i in range(0,256)]  # all states
    terminate = []
    policy = []
    attack_pos = op_state[UNIT_TYPE['base']]
    if attack_pos == []:
        valid_state = []
    else:
        opt_base_pos,dist2 = find_closest(pos,attack_pos)
        if not isNext(pos,opt_base_pos):
            action, pos1= move_to_pos_around(pos,opt_base_pos,mapp)
            while action == [] and attack_pos :
                # 不可到达
                attack_pos.remove(opt_base_pos)
                if len(attack_pos) > 0:
                    opt_base_pos, dist2= find_closest(pos,attack_pos)
                    action, pos1= move_to_pos_around(pos,opt_base_pos,mapp)
                else:
                    valid_state = []
            policy.extend(action)
        else:
            pos1 = pos
        policy.append(np.array([pos_to_idx(pos1),ACTION_TYPE['attack'],0,0,0,0,0,get_relative_pos(pos1,opt_base_pos)]))
    
    option = Option(policy,valid_state,terminate,len(policy),identifier=6)
    return [option]

def get_relative_pos(pos1,pos2):
    # for attack position parameter
    return (pos2[0]-pos1[0]+3)*7+(pos2[1]-pos1[1]+3)

def invalid_action(act, mask):
    if act[1] > 0 and act[1] < ACTION_TYPE['produce']: # move\harvest\return
        return mask[0] and mask[act[1]] 
    elif act[1] == ACTION_TYPE['produce']:# produce
        return mask[0] and mask[4] and mask[5]
    elif act[1] == ACTION_TYPE['attack']:# attack
        return mask[0] and mask[6]
    else:
        return True # NOOP
    
def invalid_attack(split_suam):
    return split_suam[0][5] == 1 # attack mask

def create_produce_worker_option(pos,my_state,num_base):
    valid_state = [i for i in range(0,256)]  # all states
    terminate = []
    policy = []
    if len(my_state[UNIT_TYPE['worker']]) >= 2*num_base:
        valid_state = []
    else:
        policy.append(np.array([pos_to_idx(pos),ACTION_TYPE['produce'],0,0,0,DIR_TO_IDX['south'],PRODUCE_TYPE['worker'],0]))
    option = Option(policy,valid_state,terminate,len(policy),identifier=2)

    return [option]

def create_return_option(pos,unit_type,base_pos_list,mapp):
    valid_state = [i for i in range(0,256)]  # all states
    terminate = []
    # policy = np.zeros((16,16),dtype=np.array) ##16*16
    policy = [] # list of array
    if len(base_pos_list) == 0 :
        valid_state = [] # option不可执行
        option = Option(policy,valid_state,terminate,len(policy),identifier=8)
        return [option]
    if unit_type == UNIT_TYPE["worker"]:
        base_pos,dis1 = find_closest(pos,base_pos_list)
        if not isNext(pos,base_pos):
            action, pos2= move_to_pos_around(pos,base_pos,mapp)
            policy.extend(action)
            while action == []:
                # 不可到达
                base_pos_list.remove(base_pos)
                if len(base_pos_list) > 0:
                    base_pos, dist1= find_closest(pos,base_pos_list)
                    action, pos2= move_to_pos_around(pos,base_pos,mapp)
                else:
                    valid_state = [] # option不可执行
                    option = Option(policy,valid_state,terminate,len(policy),identifier=8)
                    return [option]
        else:
            pos2 = pos
            policy.append(get_harvest_return_policy(pos2,base_pos,action_type=ACTION_TYPE["return"]))
    if policy == [] and unit_type == UNIT_TYPE["worker"] and valid_state != []:
        print("a")
    option = Option(policy,valid_state,terminate,len(policy),identifier=8)
    return [option]

def create_transport_option(pos, unit_type,base_pos_list, mine_pos_list,mapp):
    """
    move to mine position, harvest, move to base position , return resource.
    """
    valid_state = [i for i in range(0,256)]  # all states
    terminate = []
    # policy = np.zeros((16,16),dtype=np.array) ##16*16
    policy = [] # list of array
    if len(mine_pos_list) == 0 or len(base_pos_list) == 0 :
        valid_state = [] # option不可执行
        option = Option(policy,valid_state,terminate,len(policy),identifier=1)
        return [option]
    if unit_type == UNIT_TYPE["worker"]:
        base_pos,dis1 = find_closest(pos,base_pos_list)
        mine_pos,dis2 = find_closest(pos,mine_pos_list)
        if not isNext(pos,mine_pos):
            action, pos1= move_to_pos_around(pos,mine_pos,mapp)
            while action == []:
                # 不可到达
                mine_pos_list.remove(mine_pos)
                if len(mine_pos_list) > 0:
                    mine_pos, dist2= find_closest(pos,mine_pos_list)
                    action, pos1= move_to_pos_around(pos,mine_pos,mapp)
                else:
                    valid_state = [] # option不可执行
                    option = Option(policy,valid_state,terminate,len(policy),identifier=1)
                    return [option]
            policy.extend(action)
        else:
            pos1 = pos
        policy.append(get_harvest_return_policy(pos1,mine_pos,action_type=ACTION_TYPE["harvest"]))
        if not isNext(pos1,base_pos):
            action, pos2= move_to_pos_around(pos1,base_pos,mapp)
            policy.extend(action)
            while action == []:
                # 不可到达
                base_pos_list.remove(base_pos)
                if len(base_pos_list) > 0:
                    base_pos, dist1= find_closest(pos,base_pos_list)
                    action, pos2= move_to_pos_around(pos,base_pos,mapp)
                else:
                    valid_state = [] # option不可执行
                    option = Option(policy,valid_state,terminate,len(policy),identifier=1)
                    return [option]
        else:
            pos2 = pos
        policy.append(get_harvest_return_policy(pos2,base_pos,action_type=ACTION_TYPE["return"]))
    option = Option(policy,valid_state,terminate,len(policy),identifier=1)
    return [option]

def find_closest(pos,pos_list):
    assert len(pos_list) > 0, "find closest pos, but pos_list is empty"
    min_dis = 999
    closest_pos = pos_list[0]
    for i in pos_list:
        if distance(pos,i) < min_dis:
            closest_pos = i
            min_dis = distance(pos,i)
    return closest_pos, min_dis

def distance(pos1,pos2):
    return abs(pos1[0]-pos2[0])+abs(pos1[1]-pos2[1])

# policy
def get_harvest_return_policy(pos,mine_pos,action_type=None):
    """
        get harvest/return policy, make sure your pos is next to the mine/base pos.
    """
    if action_type == ACTION_TYPE['harvest']:
        if mine_pos[0] == pos[0]:
            if mine_pos[1]-pos[1] == 1:
                policy = np.array([pos_to_idx(pos),ACTION_TYPE['harvest'],0,DIR_TO_IDX['east'],0,0,0,0])
            elif mine_pos[1]-pos[1] == -1:
                policy = np.array([pos_to_idx(pos),ACTION_TYPE['harvest'],0,DIR_TO_IDX['west'],0,0,0,0])

        elif mine_pos[1] == pos[1]:
            if mine_pos[0]-pos[0] == 1:
                policy = np.array([pos_to_idx(pos),ACTION_TYPE['harvest'],0,DIR_TO_IDX['south'],0,0,0,0])
            elif mine_pos[0]-pos[0] == -1:
                policy = np.array([pos_to_idx(pos),ACTION_TYPE['harvest'],0,DIR_TO_IDX['north'],0,0,0,0])
        else:
            raise("your agent is too far from mine!")
    elif action_type == ACTION_TYPE['return']:
        if mine_pos[0] == pos[0]:
            if mine_pos[1]-pos[1] == 1:
                policy = np.array([pos_to_idx(pos),ACTION_TYPE['return'],0,0,DIR_TO_IDX['east'],0,0,0])
            elif mine_pos[1]-pos[1] == -1:
                policy = np.array([pos_to_idx(pos),ACTION_TYPE['return'],0,0,DIR_TO_IDX['west'],0,0,0])
            else:
                print(2)

        elif mine_pos[1] == pos[1]:
            if mine_pos[0]-pos[0] == 1:
                policy = np.array([pos_to_idx(pos),ACTION_TYPE['return'],0,0,DIR_TO_IDX['south'],0,0,0])
            elif mine_pos[0]-pos[0] == -1:
                policy = np.array([pos_to_idx(pos),ACTION_TYPE['return'],0,0,DIR_TO_IDX['north'],0,0,0])
            else:
                print(1)
        else:
            print("your agent is too far from base!")
    else:
        print("action type wrong")
    return policy


def move_to_pos_around(pos1,pos2,mapp):
    """
    from pos1 move to pos2 around
    """
    from A_star import get_move_policy
    # new_pos = copy.deepcopy(pos1)
    # policy = []
    # while not isNext(new_pos,pos2):
    #     if pos2[0]>new_pos[0]:
    #         policy.append(np.array([pos_to_idx(new_pos),ACTION_TYPE['move'],DIR_TO_IDX['south'],0,0,0,0,0]))
    #         new_pos = move(new_pos,dir=DIR_TO_IDX['south'])
    #     elif pos2[0]<new_pos[0]:
    #         policy.append(np.array([pos_to_idx(new_pos),ACTION_TYPE['move'],DIR_TO_IDX['north'],0,0,0,0,0]))
    #         new_pos = move(new_pos,dir=DIR_TO_IDX['north'])
    #     else:
    #         if pos2[1]<new_pos[1]-1:
    #             policy.append(np.array([pos_to_idx(new_pos),ACTION_TYPE['move'],DIR_TO_IDX['west'],0,0,0,0,0]))
    #             new_pos = move(new_pos,dir=DIR_TO_IDX['west'])
    #         elif pos2[1]>new_pos[1]+1:
    #             policy.append(np.array([pos_to_idx(new_pos),ACTION_TYPE['move'],DIR_TO_IDX['east'],0,0,0,0,0]))
    #             new_pos = move(new_pos,dir=DIR_TO_IDX['east'])
    if not equal(pos1,pos2):
        policy = get_move_policy(pos1,pos2,mapp)
        if len(policy)>0:
            policy = policy[:-1]
            new_pos = move(idx_to_pos(policy[-1][0]),dir=policy[-1][2])
        else:
            policy = []
            new_pos = pos1
        
    else:
        policy = []
        new_pos = pos1
    return policy,new_pos

def isNext(pos1, pos2):
    """
    if pos1 is next to pos2
    """
    if distance(pos1,pos2) < 2 and distance(pos1,pos2)>0:
        return True
    else:
        return False

def equal(pos1, pos2):
    if abs(pos1[0] - pos2[0]) + abs(pos1[1]-pos2[1]) == 0:
        return True
    else:
        return False
    
def pos_to_idx(pos):
    idx = 16*pos[0]+pos[1]
    return idx

def idx_to_pos(idx):
    pos = [idx//16, idx%16]
    return pos

def move(pos, dir=None) -> list:
    assert dir in list(DIR_TO_IDX.values())
    new_pos = copy.deepcopy(pos)
    if dir == DIR_TO_IDX['north']:
        new_pos[0] -= 1
    elif dir == DIR_TO_IDX['south']:
        new_pos[0] += 1
    elif dir == DIR_TO_IDX['west']:
        new_pos[1] -= 1
    elif dir == DIR_TO_IDX['east']:
        new_pos[1] += 1
    if inboard(new_pos):
        return new_pos
    else:
        return pos

def inboard(pos):
    if pos[0] < 16 and pos[0]>=0 and pos[1] <16 and pos[0] >=0:
        return True
    else:
        return False
    
def discounted_return(rewards,gamma):
    try:
        discounted = 0.0
        last_discount = 1.0
        for reward_set in rewards:
            gamma_mask = [gamma**t for t in range(len(reward_set))] #len(reward_set) will work if rewards is a list of lists (from planning agent)
            discounted+= np.dot(reward_set,gamma_mask) * last_discount * gamma
            last_discount = last_discount * gamma_mask[-1]
    except TypeError: # didn't work, so rewards is a list of floats - no recursion.
        gamma_mask = [gamma**t for t in range(len(rewards))]
        discounted = np.dot(rewards,gamma_mask)
    return discounted


def q_learning_update(gamma, alpha, qfunc, cur_state, action, next_state, reward):
    """
    Inputs:
        gamma: discount factor
        alpha: learning rate
        qfunc: q function (callable)
        cur_state: current state
        action: action taken opcurrent state
        next_state: next state results from taking `action` in `cur_state`
        reward: reward received from this transition
    
    Performs in-place update of q_vals table to implement one step of Q-learning
    """
    target = reward + gamma * np.max(qfunc(next_state))
    td_err = target-qfunc(cur_state)[action]
    qfunc.update_table(cur_state,qfunc(cur_state)[action] + alpha * td_err,action)#table[str(cur_state)][action] = qfunc(cur_state)[action] + alpha * td_err
    return td_err

def q_learning_update_intraoption(gamma, alpha, qfunc, states, rewards, actions):
    """Does an update to the q-table of an option based on the list of states,
       actions, and rewards obtained by following that option to termination.
    """
    td_errs = []
    T = len(rewards)
    for t in range(T):
        td_errs.append(q_learning_update(gamma, alpha, qfunc, states[t], \
            actions[t], states[t+1], rewards[t]))
    return td_errs
    
def q_learning_update_option_sequence(gamma, alpha, qfunc, states, rewards, option_index):
    """Does an update like q_learning_update, but using a sequence of states,
       actions, and rewards obtained from following an option to termination.
       USED FOR SMDP Q-LEARNING WITHOUT PLAN
    """
    td_errs = []
    T = len(rewards)
    for t in range(T):
        td_errs.append(q_learning_update(gamma, alpha, qfunc, states[t], \
            option_index, states[t+1], discounted_return(rewards[t:],gamma)))
    return td_errs

def q_learning_update_plan_options(gamma, alpha, qfunc, states, rewards, plan_option_index):
    """Does an update like q_learning_update, but using a sequence of states,
       actions, and rewards obtained from following an option to termination.
       USED FOR SMDP Q-LEARNING WITH PLAN
    """
    td_errs = []
    T = len(rewards)
    for t in range(T-1):
        td_errs.append(q_learning_update(gamma, alpha, qfunc, states[t], \
            plan_option_index, states[t+1], discounted_return(rewards[t:],gamma)))
    return td_errs

          
def greedy_eval(agent, gamma, max_steps, evals=100):
    """evaluate greedy policy w.r.t current q_vals
       max_steps:
        -> for (re)planning agent, it is the number of times the plan can be remade
        -> for smdp, it is the number of options that can be chosen.
        -> for q, it is the number of primitive actions that can be chosen.
    """
    test_env = RoomWorld()
    test_env.add_agent(agent)
    #steps = 0
    ret = 0.
    steps = 0.
    choices = 0. # number of step, option, or plan choices, depending on type
    successes = 0.
    try: # Planning Agent
        for i in range(evals):
            prev_state = test_env.reset(random_placement=True)
            done = [False]
            reward_record = []
            for s in range(max_steps):
                _ = agent.make_plan(prev_state)
                states, actions, rewards, done = test_env.step_plan(agent.sebango)
                for r in rewards:
                    reward_record.append(r)
                steps += np.sum([len(s) for s in states])
                choices += 1
                prev_state = states[-1][-1]
                if done[-1]:
                    successes += 1.
                    break
            ret += discounted_return(reward_record,gamma)
    except(AttributeError): #s-MDP Agent
        try:
            for _ in range(evals):
                prev_state = test_env.reset(random_placement=True)
                reward_record = []
                done = False
                for s in range(max_steps):
                    option = agent.pick_option_greedy_epsilon(prev_state,eps=0.0)
                    states, actions, rewards, done = test_env.step_option(option)
                    reward_record.append(rewards) # ret += np.sum(rewards)
                    prev_state = states[-1]
                    steps += len(states)
                    choices += 1
                    if done:
                        successes += 1.
                        break
                ret += discounted_return(reward_record,gamma)
        except(AttributeError): # Flat Q-learning Agent
            for i in range(evals):
                prev_state = test_env.reset(random_placement=True)
                reward_record = []
                done = False
                for s in range(max_steps):
                    action = agent.greedy_action(prev_state)
                    state, reward, done = test_env.step(action)
                    reward_record.append(reward) # ret += reward
                    prev_state = state
                    steps += 1
                    choices += 1
                    if done:
                        successes += 1.
                        break
                ret += discounted_return(reward_record,gamma)
    finally:
        return (steps/evals, choices/evals, ret/evals, successes/evals)

def switching_greedy_eval(agent, gamma, max_options, evals=100):
    """evaluate greedy policy w.r.t current q_vals with option interruption
    """
    test_env = RoomWorld()
    test_env.add_agent(agent)
    #steps = 0
    ret = 0.
    steps = 0.
    choices = 0. # number of option or plan choices, depending on type
    successes = 0.
    for _ in range(evals):
        prev_state = test_env.reset(random_placement=True)
        reward_record = []
        done = False
        for s in range(max_options):
            opt      = agent.pick_option_greedy_epsilon(prev_state, eps=0.0)
            choices += 1.
            rewards  = []
            switch   = False
            while not switch:
                action = opt.act(prev_state)
                steps += 1.
                if action is None: # option was invalid or at terminal state
                    switch = True
                    if len(rewards)==0: # count bad option choice as idle step and give R=0.
                        rewards.append(0.0)
                    reward_record.append(rewards)
                else: # option was valid
                    prev_state, re, done = test_env.step(action,agent.sebango)
                    rewards.append(re)
                    if done: # episode is done, so time to leave the option loop
                        switch = True
                        reward_record.append(rewards)
                    else: # if not done, decide whether or not to switch
                        qs = agent.q_func(prev_state)
                        if qs[opt.identifier]<np.max(qs):
                        # TODO: Add a margin so it doesn't get too trigger happy?
                            switch = True
                            reward_record.append(rewards)
            if done:
                successes += 1.
                break
        ret += discounted_return(reward_record,gamma)
    return (steps/evals, choices/evals, ret/evals, successes/evals)

def arrayify_q(q_func,walkability):
    if isinstance(q_func.table, np.ndarray):
        return q_func.table
    # Put the q-function into an array
    h,w = walkability.shape
    Q = np.zeros((h,w,q_func.num_actions))
    for k in q_func.table.keys():
        ij = k.lstrip("[ ").rstrip(" ]").split(" ")
        i  = ij[0]
        j  = ij[-1]
        Q[int(i),int(j)] = q_func.table[k]
    return Q

def plot_greedy_policy(q_array,walkability,action_directions=np.array([[1,0],[0,1],[-1,0],[0,-1]])):
    # ASSUMES THAT q_func AND walkability HAVE THE SAME DIMENSIONS ALONG AXES
    # 0 AND 1!
    h,w = walkability.shape
    Q = q_array#ify_q(q_func,walkability)
    G = np.argmax(Q,axis=2)  # table of greedy action indices (i.e., policy lookup table)
    D = np.zeros((h,w,action_directions.shape[1])) # table of greedy direction
                                                   # of motion
    for i,r in enumerate(G):
        for j,c in enumerate(r):
            if walkability[i][j]:
                D[i][j] = action_directions[c]
            else:
                D[i][j] = np.zeros_like(action_directions[0])
    x=np.linspace(0,12,13)
    x,y=np.meshgrid(x,x)
    plt.quiver(x,-y,D[:,:,0],D[:,:,1],scale_units="xy",scale=1.25) # Rooms were mapped y-down.
    plt.show()
    
    return Q,G,D


def timeStamped(fname, fmt='%Y%m%d-%H%M_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

def final_plots(walkability,q_array,hist,avg_period=100):
    l_hist, n_hist = hist.shape
    if l_hist < avg_period:
        avg_period = l_hist // 10
        print("Averaging period was too long. Reset to {}".format(avg_period))
    avg_hist       = np.zeros((l_hist-avg_period,n_hist))
    for i in range(avg_hist.shape[0]):
        avg_hist[i,:] = np.mean(hist[i:i+avg_period,:],axis=0)
    # labels
    if n_hist == 7:
        labels = ["Training Steps","Update Amount","Training Return","Test Return",
                 "Test Success Rate","Test Steps","Test choices"]
    elif n_hist == 8:
        labels = ["Training Steps","HLC Update Amount","Training Return","Test Return",
                 "Test Success Rate","Test Steps","Test Choices","LLC Update Amount"]
    else:
        print("invalid history size. plotting without labels")
        labels = [" "]*(n_hist)
    fig,axes = plt.subplots(n_hist-1,1,sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(hist[:,0],hist[:,i+1],avg_hist[:,0],avg_hist[:,i+1])
        ax.set_title(labels[i+1], fontsize=10)
    ax.set_xlabel(labels[0],fontsize=10)
    fig.tight_layout(pad=1.02,h_pad=0.0)
    plt.show()
    plt.savefig("./")
    
    try:
        Q,G,D = plot_greedy_policy(q_array, walkability)
        return Q
    except IndexError:
        print("WARNING: cannot plot policy quiverplot for more than 4 actions. Skipping.")
        
def pickle_results(obj, fname):    
    if os.path.isfile(fname):
        print("File {} already exists. Please move to avoid data loss.".format(fname))
        return "NOT SAVED"
    else:
        with open(fname,"wb") as f:
            pkl.dump(obj,f,protocol=2)
        return fname
    
def plot_and_pickle(env,ag,hist):
	# save files with check inside pickle_results
    print("Pickling data")
    filename = timeStamped("training-history.pkl")
    saved    = pickle_results(hist,filename)
    print("  --training history saved: {}".format(saved))
    filename = timeStamped("qfunc.pkl")
    Q        = arrayify_q(ag.q_func,env.walkability_map)
    saved    = pickle_results(Q,filename)
    print("  --Q-function ndarray saved: {}".format(saved))
	# Plot results
    print("Plotting results")
    final_plots(env.walkability_map,Q,hist)


def plot_room(env):
    wm = env.walkability_map
    plt.imshow(-wm,cmap="Greys")
    ax = plt.gca();
    ax = plt.gca();
    # Major ticks
    ax.set_xticks(np.arange(0, wm.shape[1], 1));
    ax.set_yticks(np.arange(0, wm.shape[0], 1));
    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, wm.shape[1]+1, 1));
    ax.set_yticklabels(np.arange(1, wm.shape[0]+1, 1));
    # Minor ticks
    ax.set_xticks(np.arange(-.5, wm.shape[1], 1), minor=True);
    ax.set_yticks(np.arange(-.5, wm.shape[0], 1), minor=True);
    # Gridlines based on minor ticks
    ax.grid(which='minor')
    plt.show()