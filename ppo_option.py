# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.spaces import MultiDiscrete
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder
# from stable_baselines3.dqn import DQN
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from rl_utils import *

if __name__ == "__main__":
    num_options = 13
    parser = argparse.ArgumentParser(description="PPO agent")
    # Common arguments
    parser.add_argument(
        "--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment"
    )
    parser.add_argument("--gym-id", type=str, default="MicrortsDefeatCoacAIShaped-v3", help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=100000000, help="total timesteps of the experiments")
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will not be enabled by default",
    )
    parser.add_argument(
        "--prod-mode",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="run the script in production mode and use wandb to log outputs",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)",
    )
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument("--n-minibatch", type=int, default=4, help="the number of mini batch")
    parser.add_argument(
        "--num-bot-envs", type=int, default=24, help="the number of bot game environment; 16 bot envs measn 16 games"
    )
    parser.add_argument(
        "--num-selfplay-envs", type=int, default=0, help="the number of self play envs; 16 self play envs means 8 games"
    )
    parser.add_argument("--num-steps", type=int, default=256, help="the number of steps per game environment")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
    parser.add_argument("--update-epochs", type=int, default=4, help="the K epochs to update the policy")
    parser.add_argument(
        "--kle-stop",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="If toggled, the policy updates will be early stopped w.r.t target-kl",
    )
    parser.add_argument(
        "--kle-rollback",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl",
    )
    parser.add_argument("--target-kl", type=float, default=0.03, help="the target-kl variable that is referred by --kl")
    parser.add_argument(
        "--gae",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Use GAE for advantage computation",
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles wheter or not to use a clipped loss for the value function, as per the paper.",
    )

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
args.num_envs = args.num_selfplay_envs + args.num_bot_envs
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, "f")
        self.eplens = np.zeros(self.num_envs, "i")
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {"r": ret, "l": eplen, "t": round(time.time() - self.tstart, 6)}
                info["episode"] = epinfo
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos


class MicroRTSStatsRecorder(VecEnvWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                raw_rewards = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                info["microrts_stats"] = dict(zip(raw_names, raw_rewards))
                self.raw_rewards[i] = []
                newinfos[i] = info
        return obs, rews, dones, newinfos


# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text(
    "hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
)
if args.prod_mode:
    import wandb

    run = wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        # sync_tensorboard=True,
        config=vars(args),
        name=experiment_name,
        monitor_gym=True,
        save_code=True,
    )
    wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    CHECKPOINT_FREQUENCY = 50

# TRY NOT TO MODIFY: seeding
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=args.num_selfplay_envs,
    num_bot_envs=args.num_bot_envs,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(args.num_bot_envs - 6)]
    + [microrts_ai.randomBiasedAI for _ in range(2)]
    + [microrts_ai.lightRushAI for _ in range(2)]
    + [microrts_ai.workerRushAI for _ in range(2)],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
envs = MicroRTSStatsRecorder(envs, args.gamma)
envs = VecMonitor(envs)
if args.capture_video:
    envs = VecVideoRecorder(
        envs, f"videos/{experiment_name}", record_video_trigger=lambda x: x % 1000000 == 0, video_length=2000
    )
# if args.prod_mode:
#     envs = VecPyTorch(
#         SubprocVecEnv([make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)], "fork"),
#         device
#     )
assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], sw=None):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.bool()
            logits = torch.where(self.masks, logits, torch.tensor(-1e8, device=device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class OptionsPool():
    def __init__(self, options, map_size=16):
        self.options        = options
        # self.num_options    = len(self.options[0])
        self.current_option = [{} for _ in range(args.num_envs)]
        self.map_size = map_size
        self.option_mask = np.zeros((args.num_envs,map_size,map_size,num_options))
        self.agent_pos = [{} for _ in range(args.num_envs)]
    
    def getOptionMask(self,obs):
    #TODO
        for i in range(args.num_envs):
            option_mask = []
            for j in range(self.map_size*self.map_size):
                if j in self.options[i].keys():
                    unit_type = self.agent_pos[i][j]
                    option_mask.extend([self.options[i][j][k].check_validity(idx_to_pos(j),unit_type,obs[i]) for k in range(num_options)])
                else:
                    option_mask.extend([False for _ in range(num_options)])
                
            self.option_mask[i] = np.array(option_mask).reshape(self.map_size,self.map_size,num_options)
        return self.option_mask

    def update_options(self, obs_tensor,action_mask,split_index,options=None):
        if options is not None:
            self.options = options
        else:
            self.options = [{} for _ in range(args.num_envs)]
            # obs = obs_tensor.cpu().numpy()
            for i in range(args.num_envs):
                agent_pos,mine_pos,base_pos,my_state,op_state,resource = obs_wrapper(obs_tensor[i])
                self.agent_pos[i] = agent_pos
                mapp = np.ones([self.map_size,self.map_size])-obs_tensor[i,:,:,13].cpu().numpy()
                for cur_state_idx, unit_type in agent_pos.items():
                    source_unit_action_mask = action_mask[i][cur_state_idx]
                    split_suam = np.split(source_unit_action_mask, split_index)
                    self.options[i][cur_state_idx] = create_options(idx_to_pos(cur_state_idx),unit_type, mine_pos,base_pos,my_state,op_state,split_suam,resource,mapp)

    def get_current_options(self,opt):
        for i in range(args.num_envs):
            for idx in self.agent_pos[i].keys():
                if idx not in self.current_option[i].keys():
                    self.current_option[i][idx] = self.options[i][idx][opt[i][idx]]
        return self.current_option
    
    def update_current_option(self,option):
        for i in range(args.num_envs):
            new_option, finished_option = check_options_termination(option[i])
            self.current_option[i] = new_option
    
def check_options_termination(options):
    finished_option = {}
    new_option = copy.deepcopy(options)
    for idx,opt in options.items():
        if opt.check_termination(idx_to_pos(idx)):
            finished_option[idx] = new_option.pop(idx)
    return new_option,finished_option

def get_action(obs, cur_option, action_mask, split_index):
    new_obs = obs.view(args.num_envs,mapsize,-1)
    acts = [[] for _ in range(args.num_envs)]
    new_option = copy.deepcopy(cur_option)
    for i in range(args.num_envs):
        for idx, opt in cur_option[i].items():
            if torch.argmax(new_obs[i,idx,21:]).item() != 0: #正在执行动作
                pass
            else:
                source_unit_action_mask = action_mask[i][idx]
                split_suam = np.split(source_unit_action_mask, split_index) # len=7, length is [6,4,4,4,4,7,49]
                if invalid_attack(split_suam):
                    opt_pos = np.argmax(split_suam[6]) # attack position 
                    act = np.array([idx,ACTION_TYPE['attack'],0,0,0,0,0,opt_pos])
                    acts[i].append(act)
                else:
                    act = opt.policy[opt.cur_action]
                    mask = [split_suam[i-1][act[i]] for i in range(1,8)]
                    # 1. 有动作 2 动作可执行 3 动作方向参数可执行
                    if invalid_action(act, mask) :
                        new_option[i][idx].cur_action += 1
                        acts[i].append(act)
                        if act[1] == ACTION_TYPE['move']:
                            new_pos = move(idx_to_pos(idx),dir=act[2])
                            new_option[i][pos_to_idx(new_pos)] = new_option[i].pop(idx)
                    else:
                        if act[1] == ACTION_TYPE['produce'] and act[6] == PRODUCE_TYPE['worker']:
                            acts[i].append(np.array([idx,ACTION_TYPE['NOOP'],0,0,0,0,0,0]))
                        else:
                            new_option[i][idx].cur_action = new_option[i][idx].num_actions # 终止option
    
    return acts,new_option

def obs_wrapper(rts_obs): ## position rts_obs 1*16*16*27
    agent_pos = {} # agent which can excecute an action now! key:pos,values:unit_type
    mine_pos = []
    base_pos = []
    my_state = {}
    op_state = {}
    num_resource = 0
    for unit_type in range(1,8): # unittype: 1-7
        my_state[unit_type] = []
        op_state[unit_type] = []
    # rts_obs_space = rts_obs.shape(-1)
    size = rts_obs.shape[1] ## 16*16*27
    for i in range(size):   ## x坐标
        for j in range(size):   ## y坐标
            hit_points = torch.argmax(rts_obs[i,j,0:5]).item()
            resource =  torch.argmax(rts_obs[i,j,5:10]).item()
            owner = torch.argmax(rts_obs[i,j,10:13]).item()
            unit_type = torch.argmax(rts_obs[i,j,13:21]).item()
            current_action = torch.argmax(rts_obs[i,j,21:27]).item()
            if owner == 1 and unit_type>0:
                my_state[unit_type].append([i,j])
                if unit_type == UNIT_TYPE['base']:
                    base_pos.append([i,j])
                    num_resource += resource
                if unit_type >= UNIT_TYPE['base'] and current_action == 0:  ## not base and resource of player1
                    agent_pos[pos_to_idx([i,j])] = unit_type
            elif owner == 2 and unit_type>0:
                op_state[unit_type].append([i,j])
            elif owner==0 and unit_type>0:
                op_state[unit_type].append([i,j])
                my_state[unit_type].append([i,j])
                if unit_type == UNIT_TYPE['resource'] and resource>0:
                    mine_pos.append([i,j])
    return agent_pos,mine_pos,base_pos,my_state,op_state,resource

class Agent(nn.Module):
    def __init__(self, mapsize=16 * 16):
        super(Agent, self).__init__()
        self.mapsize = mapsize
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 6 * 6, 256)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(256, self.mapsize * num_options), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        return self.network(x.permute((0, 3, 1, 2)))  # "bhwc" -> "bchw"

    def get_option(self, x, option=None, invalid_option_masks=None, envs=None):
        logits = self.actor(self.forward(x)) # [24,2560]
        grid_logits = logits.view(-1,num_options) #[6144,10] 
        assert invalid_option_masks is not None, "invalid_option_mask is None."
        invalid_option_masks = invalid_option_masks.view(-1, invalid_option_masks.shape[-1]).to(device)
        # invalid_option_masks [24,2560] x[24,16,16,27]
        if option is None:
            categorical = CategoricalMasked(logits=grid_logits, masks=invalid_option_masks)
            option = categorical.sample()# [6144] (24*256)
        else:
            option = option.view(-1)
            categorical = CategoricalMasked(logits=grid_logits, masks=invalid_option_masks)
        logprob = categorical.log_prob(option)#[6144]
        entropy = categorical.entropy() #[6144]
        logprob = logprob.T.view(-1, 256) #[24,256]
        entropy = entropy.T.view(-1, 256)
        option = option.T.view(-1, 256) #[24,256]
        invalid_option_masks = invalid_option_masks.view(-1, self.mapsize,num_options) # 24,256,10
        return option, logprob.sum(1), entropy.sum(1), invalid_option_masks

    def get_value(self, x):
        return self.critic(self.forward(x))


agent = Agent().to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate

# ALGO Logic: Storage for epoch data
mapsize = 16 * 16
action_space_shape = (mapsize, envs.action_space.shape[0] - 1)
invalid_action_shape = (mapsize, envs.action_space.nvec[1:].sum() + 1)
# option_space_shape = mapsize
invalid_option_shape = (mapsize,num_options)

obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + action_space_shape).to(device)
options = torch.zeros((args.num_steps, args.num_envs, mapsize)).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + invalid_action_shape).to(device)
invalid_option_masks = torch.zeros((args.num_steps, args.num_envs) + invalid_option_shape).to(device)
current_option = OptionsPool([{} for _ in range(args.num_envs)])
# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
next_obs = torch.Tensor(envs.reset()).to(device)
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size
split_index = [sum(envs.action_space.nvec.tolist()[1:a]) for a in range(2,8)]
## CRASH AND RESUME LOGIC:
starting_update = 1
from jpype.types import JArray, JInt

if args.prod_mode and wandb.run.resumed:
    starting_update = run.summary.get("charts/update") + 1
    global_step = starting_update * args.batch_size
    api = wandb.Api()
    run = api.run(f"{run.entity}/{run.project}/{run.id}")
    model = run.file("agent.pt")
    model.download(f"models/{experiment_name}/")
    agent.load_state_dict(torch.load(f"models/{experiment_name}/agent.pt", map_location=device))
    agent.eval()
    print(f"resumed at update {starting_update}")

for update in range(starting_update, num_updates + 1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]["lr"] = lrnow

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(0, args.num_steps):
        envs.render()
        
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done
        action_mask = envs.get_action_mask()  ##24*256*78
        current_option.update_options(obs[step],action_mask,split_index)
        # invalid_option_masks = current_option.getOptionMask(obs[step])
        # ALGO LOGIC: put action logic here
        t_start = time.time()
        with torch.no_grad():
            values[step] = agent.get_value(obs[step]).flatten()
            option, logproba, _, invalid_option_masks[step] = agent.get_option(obs[step],invalid_option_masks = torch.tensor(current_option.getOptionMask(obs[step])),envs=envs)
        t_end = time.time()
        # print("get_option:", t_end-t_start)
        t_start = time.time()
        options[step] = option
        logprobs[step] = logproba
        invalid_action_mask = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
        invalid_action_masks[step] = invalid_action_mask.view(-1, 256, envs.action_space.nvec[1:].sum() + 1)
        real_option = current_option.get_current_options(option)
        action, real_option = get_action(next_obs,real_option,action_mask,split_index) #[24,256,7]
        current_option.update_current_option(real_option)
        t_end = time.time()
        # print("get_action:", t_end-t_start)
        # TRY NOT TO MODIFY: execute the game and log data.
        # the real action adds the source units
        # real_action = torch.cat(
        #     [torch.stack([torch.arange(0, mapsize, device=device) for i in range(envs.num_envs)]).unsqueeze(2), action], 2
        # )

        # # at this point, the `real_action` has shape (num_envs, map_height*map_width, 8)
        # # so as to predict an action for each cell in the map; this obviously include a
        # # lot of invalid actions at cells for which no source units exist, so the rest of
        # # the code removes these invalid actions to speed things up
        # real_action = real_action.cpu().numpy()
        # valid_actions = real_action[invalid_action_masks[step][:, :, 0].bool().cpu().numpy()]
        # valid_actions_counts = invalid_action_masks[step][:, :, 0].sum(1).long().cpu().numpy()
        # java_valid_actions = []
        # valid_action_idx = 0
        # for env_idx, valid_action_count in enumerate(valid_actions_counts):
        #     java_valid_action = []
        #     for c in range(valid_action_count):
        #         java_valid_action += [JArray(JInt)(valid_actions[valid_action_idx])]
        #         valid_action_idx += 1
        #     java_valid_actions += [JArray(JArray(JInt))(java_valid_action)]
        # java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_actions)
        t_start = time.time()
        try:
            for i in range(10):
                if i == 0 :
                    next_obs, rs, ds, infos = envs.step(action)
                else:
                    next_obs, _,_,_ = envs.step([[]]*args.num_envs)
            # next_obs, rs, ds, infos = envs.step(action)
                next_obs = torch.Tensor(next_obs).to(device)
                rewards[step], next_done = torch.Tensor(rs).to(device), torch.Tensor(ds).to(device)

                for info in infos:
                    if "episode" in info.keys():
                        print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                        writer.add_scalar("charts/episode_reward", info["episode"]["r"], global_step)
                        for key in info["microrts_stats"]:
                            writer.add_scalar(f"charts/episode_reward/{key}", info["microrts_stats"][key], global_step)
                        break
        except Exception as e:
            e.printStackTrace()
            raise
        t_end = time.time()
        # print("step:", t_end-t_start)

    # bootstrap reward if not done. reached the batch limit
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)
        if args.gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = last_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            advantages = returns - values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + action_space_shape)
    b_options = options.reshape((-1,mapsize))
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_invalid_action_masks = invalid_action_masks.reshape((-1,) + invalid_action_shape)
    b_invalid_option_masks = invalid_option_masks.reshape((-1,) + invalid_option_shape)

    # Optimizaing the policy and value network
    inds = np.arange(
        args.batch_size,
    )
    for i_epoch_pi in range(args.update_epochs):
        np.random.shuffle(inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            # raise
            _, newlogproba, entropy, _ = agent.get_option(
                b_obs[minibatch_ind], b_options.long()[minibatch_ind], b_invalid_option_masks[minibatch_ind], envs
            )
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

            # Stats
            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss
            new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (new_values - b_returns[minibatch_ind]) ** 2
                v_clipped = b_values[minibatch_ind] + torch.clamp(
                    new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2)

            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

    ## CRASH AND RESUME LOGIC:
    if args.prod_mode:
        if not os.path.exists(f"models/{experiment_name}"):
            os.makedirs(f"models/{experiment_name}")
            torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"agent.pt")
        else:
            if update % CHECKPOINT_FREQUENCY == 0:
                torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("charts/update", update, global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
    writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))

envs.close()
writer.close()
