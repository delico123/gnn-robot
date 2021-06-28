import os
import random
import time

from tools.robot import Reacher
from tools.save_logger import logger

XML_DIR = './xml' # TODO

def simulate_p(N, num_iter, max_pos=1, save_idx=0, render=True, str_only=False, min_joint=2, max_joint=3):
    """
    N: num_urdf
    """

    if str_only:
        """ Save structure only, no simulation """
        joint_dirs = [item for item in os.listdir(XML_DIR) if os.path.isdir(os.path.join(XML_DIR, item))]
        for joint_dir in joint_dirs:
            # get number of joint
            num_joint = int(joint_dir[6:]) # type int for sanity check
            
            if num_joint > max_joint or num_joint < min_joint:
                continue

            Logger = logger(N, str_only, num_joint)

            Logger.save_json(f'{save_idx}-structure_only-j_{num_joint}')
        
        return

    Logger=logger(N)
    
    # Simulate
    reacher=Reacher(render=render)
    for idx in range(N):
        print('simulating reacher_{}'.format(idx))
        # reacher.load(idx) # Modified: receive urdf path explicitly (path_idx -> urdf_path)
        urdf_path = os.path.join(XML_DIR, 'reacher_'+str(idx)+'.urdf')
        reacher.load(urdf_path)

        MAX_RANGE=int(num_iter)
        Logger.dynamics=[]
        for j in range(MAX_RANGE):
            command=[]
            for i in range(reacher.NumJoints-1):
                command.append(max_pos* (random.random()-0.5))
            joint_pos,dp,next_pos=reacher.move_pose(command)
            if j%5==0:
                Logger.append_dynamics(joint_pos,command,dp,next_pos)
        Logger.save_dynamics(idx)
        reacher.close()

    # Save data
    Logger.save_json(save_idx)


""" Test code """
if __name__== '__main__':
    simulate_p(N=2, num_iter=3) # N: urdf id, num_iter: simul step