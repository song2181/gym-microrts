from matplotlib.pyplot import flag
import numpy as np

class Option():
    """Semi-MDP option class. Deterministic policy. Callable.
       ATTRIBUTES:
           - num_actions: how many actions the option policy chooses among
           - policy: the deterministic option policy in the form of an array
                     matching the environment map shape, where entries at valid
                     states match the action for that state, other entries are
                     -1
       INPUTS:
           - state (observation)
       OUTPUTS:
           - next action (assumed to be greedy.) #TODO: intra-policy training
    """
    def __init__(self, policy, valid_states, termination_conditions, num_actions=4):
        self.policy      = policy
        self.num_actions = num_actions
        self.activation  = np.array(valid_states) # activation conditions (states)
        self.termination = np.reshape(termination_conditions,(-1,2)) # (states)
        self.exploration = 0.5
        self.move_number = 4
    
    
    def act(self,state,obs):
        """The policy. Takes state (or observation) and returns action.
           This simply reads the necessary action from self.policy
           The action is applied to the agent in the arguments
        """
        if self.check_termination(state):
            return None
        else:
            roll = np.random.random()
            if roll <= self.exploration:
                valid_actions = [i for i in np.arange(self.move_number) if self.check_action_available(state,obs,i)]
                return int(np.random.choice(valid_actions))
            else:
                if int(self.policy[tuple(state)]) not in valid_actions:
                    return int(np.random.choice(valid_actions))
                else:
                    return int(self.policy[tuple(state)])
            
    
    def greedy_action(self,state):
        """Would be used if non-deterministic. Included for compatibility,
           just in case if I add q-learning.
        """
        return self.act(state)
    
    def check_action_available(self,state,obs,action):
        if action == 0:
            res = [state[0]-1,state[1]]
        elif action == 2:
            res = [state[0]+1,state[1]]
        elif action == 1:
            res = [state[0],state[1]+1]
        elif action == 3:
            res = [state[0],state[1]-1]
        if res in obs:
            return 0
        else:
            return 1
    
    def check_validity(self,state):
        """Returns boolean indicator of whether or not the state is among valid
           starting points for this option.
        """
        if type(state)==np.ndarray:
            state = state.tolist()
        return state in self.activation.tolist()
        
        
    def check_termination(self,state):
        """Returns boolean indicator of whether or not the policy is at a 
           termination state. (or not in a valid state to begin with)
        """
        if type(state)==np.ndarray:
            state = state.tolist()
        if state in self.termination.tolist() or not self.check_validity(state):
            return True
        else:
            return False
    

class Option_Q(Option):
    """This type of option stores the policy as a Q-table instead of an action
       lookup table. Use learning_test_utilities.QTable
    """
    def __init__(self, policy, valid_states, termination_conditions, num_actions=4, success_reward=1.0):
        super().__init__(policy, valid_states, termination_conditions, num_actions=num_actions)
        self.success_reward = success_reward
        
        
    def act(self,state):
        """Takes state (or observation) and returns action (argmax(Q)).
        """
        if self.check_termination(state):
            return None
        else:
            return np.argmax(self.policy(state))