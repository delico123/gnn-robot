import numpy as np
from make_urdf import make_test_urdf
from tools.robot import Reacher
import time
from parse_log import paser
x,o, command,s,dp = paser()

x = x[-2]
command = command[-2]
S = s[-2]
dp = dp[-2]
# x = [[0.0000, 0.0000],
#         [0.0000, 0.1021],
#         [0.0000, 0.2354],
#         [1.0000, 0.3029],
#         [0.0000, 0.0000],
#         [0.0000, 0.0000],
#         [0.0000, 0.0000],
#         [0.0000, 0.0000]]

# command = [[ 0.0927],
#         [ 0.1614],
#         [ 0.0470],
#         [-0.2254],
#         [ 0.0000],
#         [ 0.0000],
#         [ 0.0000],
#         [ 0.0000]]

# S = [[ 0.0609],
#         [ 0.1552],
#         [ 0.2288],
#         [ 0.1739],
#         [-0.0127],
#         [ 0.0000],
#         [ 0.0000],
#         [ 0.0000]]

# dp =[-0.0190,  0.0392]
mode = 'ik'

link = np.asarray(x)[:,1]
ee = np.asarray(x)[:,0]

n_joints = np.where(ee==1)
n_joints = n_joints[0].tolist()[0]+1
link = link[:n_joints]
make_test_urdf(n_joints, link)

joint_states = np.asarray(S)[:n_joints].reshape(n_joints)
command = np.asarray(command)[:n_joints].reshape(n_joints)

robot = Reacher(infer=True)
robot.load('./xml/reacher_infer.urdf')

if mode == 'ik':
    '''
    given dp and current joint state 
    estimate command
    '''
    robot.move_joint(joint_states)
    robot.make_maker(dp)
    print('ready')
    value = input("Please enter m to move:\n")
    if value == 'm':
        robot.move_pose(command)
        print("moved")
    time.sleep(100)

elif mode == 'fk':
    '''
    given command and current joint state 
    estimate dp
    '''
    robot.move_joint(joint_states)
    robot.make_maker(dp)
    print('ready')
    value = input("Please enter m to move:\n")
    if value == 'm':
        robot.move_pose(command)
        print("moved")
    time.sleep(100)