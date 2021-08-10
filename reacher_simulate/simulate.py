import os
import random
import time

from tools.robot import Reacher
from tools.save_logger import logger

XML_DIR = './xml' # TODO

def simulate_p(N, num_iter, max_pos=1, save_idx=0, render=True, str_only=False, min_joint=2, max_joint=7, xml_dir=XML_DIR):
    """
    N: num_urdf
    """
    XML_DIR = xml_dir

    # Simulate (after str_only)
    reacher=Reacher(render=render)
    
    joint_dirs = [item for item in os.listdir(XML_DIR) if os.path.isdir(os.path.join(XML_DIR, item))]
    for joint_dir in joint_dirs: # e.g., joint_2 in [joint_2, 3, ..]
        # get number of joint
        num_joint = int(joint_dir[6:]) # type int for sanity check
        
        if num_joint > max_joint or num_joint < min_joint:
            continue
        print(num_joint)
        Logger = logger(path_idx=N, num_joint=num_joint, xml_dir=xml_dir)
        if str_only:
            """ Save structure only, no simulation, for each joint """
            Logger.save_json(f'{save_idx}-structure_only-j_{num_joint}')
        else: # simulate then save
            for idx in range(N):
                print('simulating j{}-reacher_{}'.format(num_joint, idx))
                # reacher.load(idx) # Modified: receive urdf path explicitly (path_idx -> urdf_path)
                urdf_path = os.path.join(XML_DIR, joint_dir, 'reacher_'+str(idx)+'.urdf')
                reacher.load(urdf_path)

                MAX_RANGE=int(num_iter)
                for j in range(MAX_RANGE):
                    #time.sleep(1)
                    joint_pos, pos, collision = reacher.move_pose()
                    if collision == False:
                        Logger.append_dynamics(joint_pos, pos)
                Logger.save_dynamics(idx)
                reacher.close()

            # Save data
            Logger.save_json(f'{save_idx}-j_{num_joint}')

    # # Simulate
    # reacher=Reacher(render=render)
    # for idx in range(N):
    #     print('simulating reacher_{}'.format(idx))
    #     # reacher.load(idx) # Modified: receive urdf path explicitly (path_idx -> urdf_path)
    #     urdf_path = os.path.join(XML_DIR, 'reacher_'+str(idx)+'.urdf')
    #     reacher.load(urdf_path)

    #     MAX_RANGE=int(num_iter)
    #     Logger.dynamics=[]
    #     for j in range(MAX_RANGE):
    #         command=[]
    #         for i in range(reacher.NumJoints-1):
    #             command.append(max_vel* (random.random()-0.5))
    #         joint_pos,dp=reacher.move_pose(command)
    #         if j%5==0:
    #             Logger.append_dynamics(joint_pos,command,dp)
    #     Logger.save_dynamics(idx)
    #     reacher.close()

    # # Save data
    # Logger.save_json(save_idx)


""" Test code """
if __name__== '__main__':
    simulate_p(N=5, num_iter=200) # N: urdf id, num_iter: simul step