# Shared utilities for testing various RL schemes on the Sutton Room World
import datetime
import pickle as pkl
import os.path
import numpy as np
import matplotlib.pyplot as plt
from smdp_qlearning_algo import *

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

def create_options(landmarks):
    ## No action:-1
    # north:0
    # east:1
    # south:2
    # west:3 ##
    options = []
    ## landmarks [1,4],[4,4],[4,7]
    for lm in landmarks:
        valid_state = []
        policy = np.zeros((16,16),dtype=np.int64) ##16*16
        for i in range(0,16):
            for j in range(0,16):
                if 0 < abs(i-lm[0]) + abs(j-lm[1]) <= 3 :
                    valid_state.append([i,j])
                    if j - lm[1] < 0:
                        policy[i,j] = 1 ## move east
                    elif j - lm[1] > 0:
                        policy[i,j] = 3  ## move west
                    else:
                        if i-lm[0] < 0:
                            policy[i,j] = 2  ## move south
                        else:
                            policy[i,j] = 0  ## move north
                else:
                    policy[i,j] = -1
        options.append(Option(policy,valid_state,lm))
    return options

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