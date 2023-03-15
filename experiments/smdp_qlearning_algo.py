import numpy as np

class Agent():
    """Base agent class for interacting with RoomWorld.
    """
    def __init__(self, initial_position=[1,1]):
        self.set_position(initial_position)
        self.initial_position = self.position
        self.num_actions      = 4 #UP,DOWN,LEFT,RIGHT


    def random_action(self):
        """Random action generator. State shape is (n_samples,i,j) where i,j
           are the rows and columns of a single observation.
        """
        return np.random.randint(low=0, high=self.num_actions)


    def move(self, direction):
        """Tries moving the agent in the specified direction.
           Returns new position, which must be checked against the map
           and approved or disapproved by the environment manager.
           direction is as specified at the start of this file.
                  numpy array has axis 0 vertical, so coordinates are (-y,x).
        """
        ax0_change  = (direction % 2) * (direction - 2)
        ax1_change = ((direction+1) % 2) * (1 - direction)
        return np.array(self.get_position() + [ax0_change,ax1_change])

    
    def set_position(self, new_pos):
        self.position = np.array(new_pos)

    def get_position(self):
        return self.position


class Agent_Q(Agent):
    """Agent class using a q-function to choose actions.
       q_func should be a callable object that takes observation as input and
       outputs an array of q-values for the various actions
    """
    def __init__(self, q_func, initial_position=[1,1]):
        super().__init__(initial_position=initial_position)
        self.q_func          = q_func


    def greedy_action(self, state):
        q_values = self.q_func(state)
        return np.argmax(q_values)

    def epsilon_greedy_action(self, state, eps=0.1):
        roll = np.random.random()
        if roll <= eps:
            return self.random_action()
        else:
            return self.greedy_action(state)

class SmdpAgent_Q(Agent_Q):
    """Agent class with q-function for choosing among predefined options.
       SmdpAgent_Q takes a list of trained options that number the same as 
       the number of outputs from q_func. The Q-values are used to determine 
       which option is used whenever a option is terminated.
       No interruption is encoded in this class, but the same effect could be
       attained by adding a critic to terminate the option when another option
       is deemed more beneficial.
    """
    def __init__(self, q_func, options, initial_position=[1,1]):
        super().__init__(q_func, initial_position=initial_position)
        if not len(options)==self.q_func.num_actions:
            print("WARNING: Number of options does not match Q-table dimensions")
        self.options        = options
        self.num_options    = len(self.options)
        self.current_option = None
        for i,opt in enumerate(self.options): # label the options for q-learning
            opt.identifier = i


    def pick_option_greedy_epsilon(self, state, eps=0.0):
        """Chooses a new option to apply at state
        """
        valid_options = [i for i in np.arange(self.num_options) if self.options[i].check_validity(state)]
        all_qs        = self.q_func(state)
        valid_qs      = [all_qs[i] for i in valid_options]
        roll = np.random.random()
        if roll <= eps:
            self.current_option = np.random.choice(valid_options)
        else:
            self.current_option = valid_options[np.argmax(valid_qs)]
        return self.options[self.current_option]
        
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
        self.exploration = 0
        self.move_number = 4
    
    
    def act(self,state,obs):
        """The policy. Takes state (or observation) and returns action.
           This simply reads the necessary action from self.policy
           The action is applied to the agent in the arguments
        """
        # if self.check_termination(state):
        #     return None
        # else:
        #     return int(self.policy[tuple(state)])
        if self.check_termination(state):
            return None
        else:
            roll = np.random.random()
            valid_actions = [i for i in np.arange(self.move_number) if self.check_action_available(state,obs,i)]
            if roll <= self.exploration:
                
                return int(np.random.choice(valid_actions))
            else:
                if int(self.policy[tuple(state)]) not in valid_actions:
                    return int(np.random.choice(valid_actions))
                else:
                    return int(self.policy[tuple(state)])
            
    def check_action_available(self,state,obs,action):
        if action == 0:
            res = [state[0]-1,state[1]]
        elif action == 2:
            res = [state[0]+1,state[1]]
        elif action == 1:
            res = [state[0],state[1]+1]
        elif action == 3:
            res = [state[0],state[1]-1]
        if res in [list(i) for i in obs]:
            return 0
        else:
            return 1

    def greedy_action(self,state):
        """Would be used if non-deterministic. Included for compatibility,
           just in case if I add q-learning.
        """
        return self.act(state)
    
    
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

