import numpy as np
from numpy.random import choice
# if you want to record videos, install stable-baselines3 and use its `VecVideoRecorder`
# from stable_baselines3.common.vec_env import VecVideoRecorder

from gym_microrts_old import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
import rl_utils as utils

envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(1)],
    map_path="maps/16x16/basesWorkers16x16.xml",
    # map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
# envs = VecVideoRecorder(envs, 'videos', record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sample(logits):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    if sum(logits) == 0: return 0
    return choice(range(len(logits)), p=logits/sum(logits))

envs.action_space.seed(0)
envs.reset()
# print(envs.action_plane_space.nvec)
nvec = envs.action_space.nvec

def sample(logits):
    return np.array(
        [choice(range(len(item)), p=softmax(item)) for item in logits]
    ).reshape(-1, 1)

for i in range(10000):
    envs.render()
    print(i)
    # TODO: this numpy's `sample` function is very very slow.
    # PyTorch's `sample` function is much much faster,
    # but we want to remove PyTorch as a core dependency...
    action_mask = envs.get_action_mask()
    action_mask = action_mask.reshape(-1, action_mask.shape[-1])
    pos = 17
    # action = np.concatenate(
    #     (   
    #         pos,
    #         sample(action_mask[pos, 0:6]),  # action type
    #         sample(action_mask[pos, 6:10]),  # move parameter
    #         sample(action_mask[pos, 10:14]),  # harvest parameter
    #         sample(action_mask[pos, 14:18]),  # return parameter
    #         sample(action_mask[pos, 18:22]),  # produce_direction parameter
    #         sample(action_mask[pos, 22:29]),  # produce_unit_type parameter
    #         # attack_target parameter
    #         sample(action_mask[pos, 29 : sum(envs.action_space.nvec[1:])]),
    #     ),
    #     axis=1,
    # )
    # doing the following could result in invalid actions
    # action = np.array([envs.action_space.sample()])
    # action[1*16+1][0]=utils.ACTION_TYPE['harvest']
    # # action[1*16+1][0]=utils.ACTION_TYPE['move']
    # # 1-move,2-harvest,3-return,4-procduce direction,5-produce type,6-attack position
    # action[1*16+1][2]=utils.DIR_TO_IDX['west']
    # action[1*16+1][1]=utils.DIR_TO_IDX['south']

    # action[1*16+1][5]= utils.PRODUCE_TYPE['barrack']
    action = np.array([[17,utils.ACTION_TYPE['harvest'],utils.DIR_TO_IDX['south'],utils.DIR_TO_IDX['west'],0,0,0,0],
                       [2*16+2,utils.ACTION_TYPE['produce'],0,0,0,utils.DIR_TO_IDX['south'],utils.PRODUCE_TYPE['worker'],0]])
    for i in range(20):
        next_obs, reward, done, info = envs.step([action])
    envs.render()
    action = np.array([[17,utils.ACTION_TYPE['move'],utils.DIR_TO_IDX['south'],utils.DIR_TO_IDX['west'],0,0,0,0],
                       [2*16+2,utils.ACTION_TYPE['produce'],0,0,0,utils.DIR_TO_IDX['south'],utils.PRODUCE_TYPE['worker'],0]])
    for i in range(10):
        next_obs, reward, done, info = envs.step([action])
    envs.render()
    action = np.array([[2*16+1,utils.ACTION_TYPE['return'],utils.DIR_TO_IDX['south'],0,utils.DIR_TO_IDX['east'],0,0,0],
                       [2*16+2,utils.ACTION_TYPE['produce'],0,0,0,utils.DIR_TO_IDX['south'],utils.PRODUCE_TYPE['worker'],0]])
    for i in range(20):
        next_obs, reward, done, info = envs.step([action])
    envs.render()
    # act_type = [0,1,0,0,0,0]  ## move
    # move_dir = [0,0,0,0]      ## move parameter
    # action = np.zeros(shape=(1,256,78),dtype=np.int64)
    # move_dir[1] = 1
    # action[0][17] = act_type + move_dir + [0]*68


    # print(action[0][17*7:18*7])  ## worker
    # action[0][17*7:18*7] = [1,1,0,0,0,0,0]
    # print(action[0][34*7:35*7])  ## base
    # next_obs, reward, done, info = envs.step(action)
envs.close()