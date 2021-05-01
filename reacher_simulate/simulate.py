from tools.robot import Reacher
import random
import time
from tools.save_logger import logger

def simulate_p(N,num_iter,max_vel=1,save_idx=0,render=True):
    Logger=logger(N)
    reacher=Reacher(render=render)
    for idx in range(N):
        print('simulating reacher_{}'.format(idx))
        reacher.load(idx)
        MAX_RAGNE=int(num_iter)
        Logger.dynamics=[]
        for _ in range(MAX_RAGNE):
            command=[]
            for i in range(reacher.NumJoints-1):
                command.append(max_vel* (random.random()-0.5))
            joint_pos,dp=reacher.move_pose(command)
            Logger.append_dynamics(joint_pos,command,dp)
        Logger.save_dynamics(idx)
        reacher.close()
    Logger.save_json(save_idx)

if __name__== '__main__':
    simulate(2)