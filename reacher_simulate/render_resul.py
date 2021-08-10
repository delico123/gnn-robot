import os
import time
import pickle

import numpy as np
import pandas as pd

from make_urdf import make_test_urdf
from tools.robot import Reacher

# log_dir = "./log/latent"
log_dir = "./render_resource"

def simul_result_struc(robot_pkl='./result_data_struc.pkl'):
    robot = Reacher(render=True)
    n_jointss, links, motions = pickle.load(open(os.path.join(log_dir, robot_pkl), 'rb'))

    for link, n_joints, motion in zip(links, n_jointss, motions):
        link = np.asarray(link)[1:,1]

        make_test_urdf(n_joints, link)

        robot.load('./xml/reacher_infer.urdf')

        # TODO: motion pos. note: rstruc case, motion <- None
        # break

    # robot.close()
    time.sleep(100)


def test():
    import pybullet as p
    p.connect(p.GUI)
    p.loadURDF('./xml/reacher_infer.urdf')
    time.sleep(10)



def df2robot(rid, df='pretrain-rstruc-tree-df.pkl', robot_pkl='result_data_struc.pkl'):
    df = pickle.load(open(os.path.join(log_dir, df), "rb"))
    row = df.loc[rid]
    
    n_jointss = [row['n_joints'], row['n_joints']]
    links = [row['s_targ'], row['s_pred']]
    motions = [row['m_targ'], row['m_pred']]

    pickle.dump((n_jointss, links, motions), open(os.path.join(log_dir, robot_pkl), 'wb'))


def latent_pkl2df(pkl_path="pretrain-rstruc-tree", epoch_idx=-1, mode="rstruc"): # epoch_idx 0: first, -1: last latent
    # Load latent

    assert(f"-{mode}" in pkl_path)
    latent_pkl = pickle.load(open(os.path.join(log_dir, f"{pkl_path}.pkl"), "rb"))
    data = latent_pkl[epoch_idx]['val_latent']
    del latent_pkl

    # NOTE:
    # rstruc: [num_node, data.x, output, z] 
    #          == [_, structure, pred structure, struc latent]
    # rmotion: [num_node, data_set[0].x, data.x, output, z] 
    #          == [_, structure, joint pos, pred joint pos, motion latent]
    # train: [num_node, d_struc.x, d_motion.s, d_motion.p, d_motion.s, out_f, out_i, z_struc, z_motion]
    #          == [_, struc, jpos, eepos, .., pred eepos, pred jpos, struct latent, motion latent]

    if mode == "rstruc":
        n_joints = np.array([dd[0].numpy() - 1 for dd in data]) # n_joints (num_node = n_joints + 1)
        df = pd.DataFrame(n_joints, columns=['n_joints'])
        
        df['s_targ'] = list([dd[1].numpy() for dd in data])
        df['s_pred'] = list([dd[2].numpy() for dd in data])

        df['m_targ'] = None
        df['m_pred'] = None

    elif mode == "rmotion":
        n_joints = np.array([dd[0].numpy() - 1 for dd in data]) # NOTE: n_joints (num_node = n_joints + 1)
        df = pd.DataFrame(n_joints, columns=['n_joints'])
        
        df['s_targ'] = list([dd[1].numpy() for dd in data])
        df['s_pred'] = df['s_targ']

        df['m_targ'] = list([dd[2].numpy() for dd in data])
        df['m_pred'] = list([dd[3].numpy() for dd in data])

    else:
        raise NotImplementedError

    df['index_col'] = df.index

    pickle.dump(df, open(os.path.join(log_dir, f"{pkl_path}-df.pkl"), 'wb'))


    
if __name__== '__main__':
    latent_pkl2df()
    df2robot(111, robot_pkl='test-result_struc.pkl')
    simul_result_struc(robot_pkl='test-result_struc.pkl')

    # test()