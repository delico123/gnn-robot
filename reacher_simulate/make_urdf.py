import os
import random

import urdfpy

from tools.get_urdf_info import get_base,get_ee,get_link_joint

XML_DIR = './xml'

def make_reacher_urdf(save_index, min_njoint=2, max_njoint=3, MIN=0.1, MAX=0.4, str_only=False):
    """
    str_only: if true, fix joint # as min_njoint (for structure reconstruction task)
    """

    num_joint = min_njoint if str_only else random.randint(min_njoint, max_njoint)

    links=[]
    joints=[]
    l,j=get_base()
    links.append(l)
    joints.append(j)
    for i in range(num_joint):
        random_length=MIN+(MAX-MIN)*random.random()
        l,j=get_link_joint(random_length,i)
        links.append(l)
        joints.append(j)
    temp=get_ee(num_joint)
    links.append(temp)
    #for i,joint1 in enumerate(robot.joints):
    robot=urdfpy.URDF('pybullet_reacher_robot',links,joints)

    if str_only:
        joint_dir = os.path.join(XML_DIR, 'joint_{}'.format(num_joint))
        if not os.path.exists(joint_dir):
            os.makedirs(joint_dir)
        robot.save(os.path.join(joint_dir, 'reacher_'+str(save_index)+'.urdf'))
        if num_joint < max_njoint:
            make_reacher_urdf(save_index, num_joint+1, max_njoint, MIN, MAX, str_only)
    else:
        robot.save(os.path.join(XML_DIR, 'reacher_'+str(save_index)+'.urdf'))
    
def make_test_urdf(num_joint,link):
    links=[]
    joints=[]
    l,j=get_base()
    links.append(l)
    joints.append(j)
    for i in range(num_joint):
        l,j=get_link_joint(link[i],i)
        links.append(l)
        joints.append(j)
    temp=get_ee(num_joint)
    links.append(temp)
    #for i,joint1 in enumerate(robot.joints):
    robot=urdfpy.URDF('pybullet_reacher_robot',links,joints)
    robot.save(os.path.join(XML_DIR, 'reacher_infer.urdf'))

""" Test code """

if __name__=='__main__':
    NUM=2
    for i in range(NUM):
        make_reacher_urdf(i,0.1,0.4)